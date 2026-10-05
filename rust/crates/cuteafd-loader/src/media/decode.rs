use super::{exif, invalid, MediaError, Result, RgbImage};
use image::{DynamicImage, ImageDecoder, ImageReader, Limits};
use serde::{Deserialize, Serialize};
use std::io::Cursor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlphaPolicy {
    Drop,
    White,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodePolicy {
    pub exif_transpose: bool,
    pub alpha: AlphaPolicy,
}
impl Default for DecodePolicy {
    fn default() -> Self {
        Self {
            exif_transpose: true,
            alpha: AlphaPolicy::Drop,
        }
    }
}
const MAX_PIXELS: u64 = 64 << 20;
fn dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(invalid(
            "image dimensions must be positive and at most 64 megapixels",
        ));
    }
    Ok(())
}

pub fn decode(encoded: &[u8], policy: DecodePolicy) -> Result<RgbImage> {
    decode_bounded(encoded, policy, usize::MAX)
}
/// Check the request's remaining RGB budget from the header before allocation.
pub fn decode_bounded(encoded: &[u8], policy: DecodePolicy, rgb_budget: usize) -> Result<RgbImage> {
    let admit = |width, height| -> Result<()> {
        dimensions(width, height)?;
        if u64::from(width) * u64::from(height) * 3 > rgb_budget as u64 {
            return Err(invalid("decoded images exceed request byte limit"));
        }
        Ok(())
    };
    if encoded.is_empty() || encoded.len() > 32 << 20 {
        return Err(invalid("encoded image must be 1 byte through 32 MiB"));
    }
    let err = |e: String| MediaError::Decode(e);
    let (rgb, orientation) = if encoded.starts_with(&[0xff, 0xd8]) {
        // Same libjpeg-turbo settings and CMYK conversion as the proven V4.1 path.
        let mut decoder = turbojpeg::Decompressor::new().map_err(|e| err(e.to_string()))?;
        let header = decoder
            .read_header(encoded)
            .map_err(|e| err(e.to_string()))?;
        let (width, height) = (
            u32::try_from(header.width).map_err(|_| invalid("JPEG width overflow"))?,
            u32::try_from(header.height).map_err(|_| invalid("JPEG height overflow"))?,
        );
        admit(width, height)?;
        let cmyk = matches!(
            header.colorspace,
            turbojpeg::Colorspace::CMYK | turbojpeg::Colorspace::YCCK
        );
        let channels = if cmyk { 4 } else { 3 };
        let mut pixels = vec![0; header.width * header.height * channels];
        decoder
            .set_fast_upsample(false)
            .map_err(|e| err(e.to_string()))?;
        decoder
            .decompress(
                encoded,
                turbojpeg::Image {
                    pixels: pixels.as_mut_slice(),
                    width: header.width,
                    height: header.height,
                    pitch: header.width * channels,
                    format: if cmyk {
                        turbojpeg::PixelFormat::CMYK
                    } else {
                        turbojpeg::PixelFormat::RGB
                    },
                },
            )
            .map_err(|e| err(e.to_string()))?;
        let data = if cmyk {
            pixels
                .chunks_exact(4)
                .flat_map(|p| {
                    (0..3).map(move |c| ((u16::from(p[c]) * u16::from(p[3]) + 127) / 255) as u8)
                })
                .collect()
        } else {
            pixels
        };
        (
            RgbImage {
                width,
                height,
                data,
            },
            jpeg_orientation(encoded),
        )
    } else {
        let mut reader = ImageReader::new(Cursor::new(encoded))
            .with_guessed_format()
            .map_err(|e| err(e.to_string()))?;
        let mut limits = Limits::default();
        limits.max_alloc = Some(512 << 20);
        reader.limits(limits);
        let mut decoder = reader.into_decoder().map_err(|e| err(e.to_string()))?;
        let (width, height) = decoder.dimensions();
        admit(width, height)?;
        let metadata = if policy.exif_transpose {
            decoder.exif_metadata().map_err(|e| err(e.to_string()))?
        } else {
            None
        };
        let orientation = exif::orientation(metadata.as_deref(), None);
        let decoded = DynamicImage::from_decoder(decoder).map_err(|e| err(e.to_string()))?;
        let decoded = match decoded {
            // Pillow truncates multi-channel PNG16 samples to their high byte.
            DynamicImage::ImageRgb16(rgb) => DynamicImage::ImageRgb8(
                image::RgbImage::from_raw(
                    width,
                    height,
                    rgb.into_raw().into_iter().map(|v| (v >> 8) as u8).collect(),
                )
                .expect("decoded extent"),
            ),
            DynamicImage::ImageRgba16(rgba) => DynamicImage::ImageRgba8(
                image::RgbaImage::from_raw(
                    width,
                    height,
                    rgba.into_raw()
                        .into_iter()
                        .map(|v| (v >> 8) as u8)
                        .collect(),
                )
                .expect("decoded extent"),
            ),
            DynamicImage::ImageLumaA16(gray) => DynamicImage::ImageLumaA8(
                image::GrayAlphaImage::from_raw(
                    width,
                    height,
                    gray.into_raw()
                        .into_iter()
                        .map(|v| (v >> 8) as u8)
                        .collect(),
                )
                .expect("decoded extent"),
            ),
            other => other,
        };
        let data = match decoded {
            // Pillow's integer-mode 16-bit grayscale conversion clamps, not scales.
            DynamicImage::ImageLuma16(gray) => gray
                .into_raw()
                .into_iter()
                .flat_map(|v| [v.min(255) as u8; 3])
                .collect(),
            other if policy.alpha == AlphaPolicy::Drop => other.to_rgb8().into_raw(),
            other => other
                .to_rgba8()
                .pixels()
                .flat_map(|p| {
                    let a = u32::from(p[3]);
                    (0..3).map(move |c| {
                        // Pillow alpha_composite's integer half-up result over opaque white.
                        ((u32::from(p[c]) * a + 255 * (255 - a) + 127) / 255) as u8
                    })
                })
                .collect(),
        };
        (
            RgbImage {
                width,
                height,
                data,
            },
            orientation,
        )
    };
    Ok(if policy.exif_transpose {
        exif::apply(rgb, orientation)
    } else {
        rgb
    })
}

fn jpeg_orientation(bytes: &[u8]) -> i64 {
    let mut offset = 2usize;
    let (mut metadata, mut xmp) = (None, None);
    while offset < bytes.len() {
        if bytes[offset] != 0xff {
            break;
        }
        while bytes.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let Some(&marker) = bytes.get(offset) else {
            break;
        };
        offset += 1;
        if matches!(marker, 0xda | 0xd9) {
            break;
        }
        if marker == 0x01 || (0xd0..=0xd7).contains(&marker) {
            continue;
        }
        let Some(size) = bytes.get(offset..offset + 2) else {
            break;
        };
        let length = usize::from(u16::from_be_bytes([size[0], size[1]]));
        if length < 2 {
            break;
        }
        let Some(payload) = bytes.get(offset + 2..offset + length) else {
            break;
        };
        if marker == 0xe1 {
            if payload.starts_with(b"Exif\0\0") {
                metadata = Some(payload);
            }
            if let Some(value) = payload.strip_prefix(b"http://ns.adobe.com/xap/1.0/\0") {
                xmp = Some(value);
            }
        }
        offset += length;
    }
    exif::orientation(metadata, xmp)
}
