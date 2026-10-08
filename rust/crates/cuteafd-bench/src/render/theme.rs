//! Colors of the console (ds41rt aesthetic) and of scary mode, which every
//! view and export of a report whose quality gate failed uses.

#[derive(Debug, Clone, Copy)]
pub struct Theme {
    pub scary: bool,
    pub bg: &'static str,
    pub panel: &'static str,
    pub panel2: &'static str,
    pub well: &'static str,
    pub line: &'static str,
    pub line2: &'static str,
    pub ink: &'static str,
    pub ink2: &'static str,
    pub muted: &'static str,
    pub good: &'static str,
    pub warn: &'static str,
    pub bad: &'static str,
    /// Series colors in order (content types, configurations, ...).
    pub series: [&'static str; 8],
    /// Concurrent code headline, distinct from the C1 content bars and prefill.
    pub concurrent: &'static str,
    /// The two ends of the brand gradient (RTX blue, Spark violet).
    pub accent: (&'static str, &'static str),
}

pub const NORMAL: Theme = Theme {
    scary: false,
    bg: "#05070b",
    panel: "#0a0e15",
    panel2: "#0e141d",
    well: "#070a10",
    line: "#172030",
    line2: "#22304a",
    ink: "#e8eef8",
    ink2: "#9dabc1",
    muted: "#5f6e86",
    good: "#3ddc84",
    warn: "#ffb547",
    bad: "#ff5d73",
    series: ["#4cc3ff", "#2ee6a6", "#ffb547", "#a27bff", "#ff5d73", "#c08bff", "#7ee0ff", "#f5d76e"],
    concurrent: "#d66086",
    accent: ("#4cc3ff", "#c08bff"),
};

pub const SCARY: Theme = Theme {
    scary: true,
    bg: "#0b0405",
    panel: "#150709",
    panel2: "#1c0a0d",
    well: "#0e0506",
    line: "#3a1318",
    line2: "#5a1a22",
    ink: "#ffe9e9",
    ink2: "#e3a3a3",
    muted: "#94595d",
    good: "#ffb020",
    warn: "#ffb020",
    bad: "#ff3b4e",
    series: ["#ff3b4e", "#ffb020", "#ff7a2f", "#ffd166", "#e0365a", "#ff9f6e", "#ff5d73", "#f5a524"],
    concurrent: "#d66086",
    accent: ("#ff3b4e", "#ffb020"),
};

impl Theme {
    pub fn for_report(failed: bool) -> Self {
        if failed { SCARY } else { NORMAL }
    }

    /// Color of a check status.
    pub fn status(&self, status: crate::report::CheckStatus) -> &'static str {
        use crate::report::CheckStatus::*;
        match status {
            Pass => self.good,
            Fail => self.bad,
            Info => self.series[0],
            Skipped | Unsupported | Pending => self.muted,
        }
    }

    /// `<defs>` every themed document carries: the brand gradient, the
    /// hazard-stripe pattern and the background glows.
    pub fn defs(&self) -> String {
        let (a, b) = self.accent;
        format!(concat!(
            r##"<linearGradient id="accent" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{a}"/><stop offset="1" stop-color="{b}"/></linearGradient>"##,
            r##"<pattern id="hazard" width="16" height="16" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="16" height="16" fill="#120405"/><rect width="8" height="16" fill="{w}"/></pattern>"##,
            r##"<radialGradient id="glow-a" cx="0.12" cy="-0.1" r="0.7"><stop offset="0" stop-color="{a}" stop-opacity="0.10"/><stop offset="1" stop-color="{a}" stop-opacity="0"/></radialGradient>"##,
            r##"<radialGradient id="glow-b" cx="1" cy="0" r="0.6"><stop offset="0" stop-color="{b}" stop-opacity="0.09"/><stop offset="1" stop-color="{b}" stop-opacity="0"/></radialGradient>"##),
            a = a, b = b, w = self.warn)
    }
}
