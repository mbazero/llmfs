use anyhow::Result;

pub const CORPUS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/the-verdict.txt");

pub fn read_corpus() -> Result<String> {
    let bytes = std::fs::read(CORPUS_PATH)?;
    Ok(String::from_utf8(bytes)?)
}
