//! Deterministic prompt material: unique nonces (so no request reuses a cached
//! prefix) and pseudo-prose filler of a chosen length.

/// A short unique tag for one request.
pub fn nonce() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..10].to_string()
}

const WORDS: [&str; 96] = ["river", "lantern", "harbor", "orchard", "signal", "copper", "meadow", "engine",
    "letter", "window", "valley", "thunder", "garden", "market", "silver", "ladder", "bridge", "candle", "forest",
    "pocket", "anchor", "pepper", "violin", "marble", "rocket", "saddle", "tunnel", "pillow", "basket", "mirror",
    "carpet", "falcon", "island", "jacket", "kettle", "magnet", "needle", "oyster", "parrot", "quartz", "ribbon",
    "spiral", "teapot", "umbrella", "velvet", "walnut", "yellow", "zephyr", "amber", "breeze", "castle", "dragon",
    "ember", "fossil", "glacier", "hollow", "ivory", "jungle", "kernel", "lagoon", "monsoon", "nectar", "oracle",
    "prism", "quiver", "raven", "summit", "timber", "utopia", "vortex", "willow", "xylem", "yonder", "zenith",
    "atlas", "beacon", "canyon", "delta", "eclipse", "fjord", "granite", "horizon", "inlet", "jasper", "kelp",
    "lumen", "meteor", "nebula", "obsidian", "plateau", "quarry", "reef", "sierra", "tundra", "upland", "verdant"];
const VERBS: [&str; 24] = ["carries", "follows", "measures", "remembers", "answers", "gathers", "shelters",
    "outlines", "balances", "crosses", "echoes", "frames", "guides", "holds", "joins", "keeps", "lifts", "marks",
    "names", "opens", "paints", "quiets", "reaches", "shapes"];

/// Roughly `words` words of grammatical nonsense, the same for the same seed.
pub fn filler(seed: u64, words: usize) -> String {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1;
    let mut next = |n: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };
    let mut out = String::with_capacity(words * 8);
    let mut count = 0;
    while count < words {
        let length = 6 + next(8);
        let mut sentence = Vec::with_capacity(length);
        for i in 0..length {
            sentence.push(if i == 2 || i == length / 2 + 2 { VERBS[next(VERBS.len())] } else { WORDS[next(WORDS.len())] });
        }
        let mut s = sentence.join(" ");
        if let Some(first) = s.get(0..1) {
            s.replace_range(0..1, &first.to_uppercase());
        }
        out.push_str(&s);
        out.push_str(if next(5) == 0 { ".\n" } else { ". " });
        count += length;
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn filler_is_deterministic_and_sized() {
        let a = super::filler(7, 500);
        assert_eq!(a, super::filler(7, 500));
        assert_ne!(a, super::filler(8, 500));
        let words = a.split_whitespace().count();
        assert!((500..520).contains(&words), "{words}");
        assert_ne!(super::nonce(), super::nonce());
    }
}
