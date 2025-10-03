use anyhow::Result;
use tokenizer::util::CORPUS_PATH;

const THE_VERDICT_URL: &str = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";

fn main() -> Result<()> {
    let resp = ureq::get(THE_VERDICT_URL).call()?;
    let bytes = resp.into_body().read_to_vec()?;
    std::fs::write(CORPUS_PATH, &bytes)?;
    println!("Wrote file to {CORPUS_PATH}");
    Ok(())
}
