//! SVG to PNG with resvg and the embedded DejaVu Sans Mono (no system fonts,
//! so a PNG looks the same on every host).
use anyhow::{Context, Result};
use std::sync::{Arc, OnceLock};

static REGULAR: &[u8] = include_bytes!("../../assets/fonts/DejaVuSansMono.ttf");
static BOLD: &[u8] = include_bytes!("../../assets/fonts/DejaVuSansMono-Bold.ttf");

fn fonts() -> Arc<resvg::usvg::fontdb::Database> {
    static DB: OnceLock<Arc<resvg::usvg::fontdb::Database>> = OnceLock::new();
    DB.get_or_init(|| {
        let mut db = resvg::usvg::fontdb::Database::new();
        db.load_font_data(REGULAR.to_vec());
        db.load_font_data(BOLD.to_vec());
        db.set_monospace_family("DejaVu Sans Mono");
        db.set_sans_serif_family("DejaVu Sans Mono");
        db.set_serif_family("DejaVu Sans Mono");
        Arc::new(db)
    }).clone()
}

/// Rasterizes `svg` at `scale` (1.0: its own pixel size).
pub fn png(svg: &str, scale: f32) -> Result<Vec<u8>> {
    let options = resvg::usvg::Options { fontdb: fonts(), font_family: "DejaVu Sans Mono".into(),
        ..resvg::usvg::Options::default() };
    let tree = resvg::usvg::Tree::from_str(svg, &options).context("parsing the SVG")?;
    let size = tree.size().to_int_size().scale_by(scale).context("image size")?;
    let mut pixmap = resvg::tiny_skia::Pixmap::new(size.width(), size.height()).context("pixmap")?;
    resvg::render(&tree, resvg::tiny_skia::Transform::from_scale(scale, scale), &mut pixmap.as_mut());
    pixmap.encode_png().context("encoding the PNG")
}
