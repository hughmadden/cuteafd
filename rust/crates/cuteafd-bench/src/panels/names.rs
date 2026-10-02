//! Short names for dataset items whose labels are sentence prompts: charts
//! show the name, the tooltip and the table carry the full prompt.

/// (substring of the prompt, name), first match wins.
const NAMES: [(&str, &str); 37] = [
    // Math word problems.
    ("pencils in each", "pencil-boxes"),
    ("arithmetic sequence", "arith-series"),
    ("are divisible by", "divisible-count"),
    ("rectangle is", "rect-area"),
    ("A ticket costs", "ticket-cost"),
    // Reasoning puzzles.
    ("What is the remainder when", "modpow"),
    ("positive divisors does", "divisor-count"),
    ("sum of floor(", "floor-sum"),
    ("stand in a circle", "josephus"),
    ("digits summing to", "digit-sum"),
    ("divisible by neither 5 nor 7", "not-5-or-7"),
    ("sum of all positive divisors of 360", "divisor-sum-360"),
    ("2^2026", "pow2-mod-1000"),
    ("trailing zeros", "zeros-1000!"),
    ("lattice points", "lattice-r10"),
    ("summing to exactly 39", "subset-sum-39"),
    // Instruction following.
    ("haiku about rain", "lowercase-haiku"),
    ("exactly 5 fruits", "five-bullets"),
    ("vaccines work", "min-150-words"),
    ("Describe the moon", "max-40-words"),
    ("without using any commas", "no-commas"),
    ("double quotation marks", "quoted"),
    ("JSON object with keys", "json-object"),
    ("anything else I can help with", "end-phrase"),
    ("'galaxy' and 'teapot'", "keywords"),
    ("exactly 3 paragraphs", "three-paragraphs"),
    ("ALL CAPITAL LETTERS", "all-caps"),
    ("Explain recursion", "two-highlights"),
    ("word 'Absolutely'", "starts-with"),
    ("letter 'z'", "six-zs"),
    ("double angular brackets", "title-brackets"),
    ("6 asterisks", "two-names"),
    ("P.S.", "postscript"),
    ("'good' or 'bad'", "forbidden-words"),
    ("word 'river'", "river-x3"),
    ("exactly two sentences", "two-sentences"),
    ("Write a Python function", "python-function"),
];

/// The short name of a prompt: a known item's, else its first words, slugged.
pub fn short_name(prompt: &str) -> String {
    if let Some((_, name)) = NAMES.iter().find(|(needle, _)| prompt.contains(needle)) {
        return name.to_string();
    }
    let words: Vec<String> = prompt.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).take(3)
        .map(str::to_lowercase).collect();
    if words.is_empty() { "item".into() } else { words.join("-") }
}

#[cfg(test)]
mod tests {
    #[test]
    fn known_prompts_get_their_names() {
        assert_eq!(super::short_name("A rectangle is 10 m long and 3 m wide."), "rect-area");
        assert_eq!(super::short_name("Find the remainder when 2^2026 is divided by 1000."), "pow2-mod-1000");
        assert_eq!(super::short_name("Something else entirely here"), "something-else-entirely");
    }
}
