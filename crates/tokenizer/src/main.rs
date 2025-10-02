use anyhow::Result;
use indexmap::IndexMap;
use itertools::Itertools;
use regex::Regex;

const TEXT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/the-verdict.txt");

fn main() -> Result<()> {
    let text: String = std::fs::read(TEXT_PATH)?.try_into()?;
    let re = Regex::new(r#"([,.:;?_!"()\']|--|\s)"#)?;

    // TODO: Don't discard delimiters
    let vocab: IndexMap<_, _> = re
        .split(&text)
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() { None } else { Some(s) }
        })
        .unique()
        .sorted()
        .zip(0usize..)
        .collect();

    dbg!(vocab.iter().take(10).collect::<Vec<_>>());

    Ok(())
}
