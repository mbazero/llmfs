use anyhow::Result;
use indexmap::IndexMap;
use itertools::Itertools;

use crate::splitter::Splitter;

const TEXT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/the-verdict.txt");

pub mod splitter;
pub mod tokenizer;

fn main() -> Result<()> {
    let text: String = std::fs::read(TEXT_PATH)?.try_into()?;
    let splitter = Splitter::default();

    let vocab: IndexMap<_, _> = splitter
        .split_filter(&text)
        .unique()
        .sorted()
        .zip(0usize..)
        .collect();

    for (i, s) in vocab.iter().take(10) {
        println!("{s}: {i}");
    }

    Ok(())
}
