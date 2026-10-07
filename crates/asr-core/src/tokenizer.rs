use anyhow::{Context, Result, ensure};
use std::path::Path;
use tokenizers::{
    AddedToken, SplitDelimiterBehavior, Tokenizer,
    models::bpe::BPE,
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
};

pub const ASR_TEXT: u32 = 151704;
pub const STOP_TOKENS: [u32; 2] = [151645, 151643];
pub const PREFIX: [u32; 9] = [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669];
pub const SUFFIX: [u32; 6] = [151670, 151645, 198, 151644, 77091, 198];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Language {
    #[default]
    Auto,
    Chinese,
    English,
}

pub struct AsrTokenizer {
    tokenizer: Tokenizer,
}

fn tokenizer_error(error: Box<dyn std::error::Error + Send + Sync>) -> anyhow::Error {
    anyhow::anyhow!("{error}")
}

impl AsrTokenizer {
    pub fn load(directory: &Path) -> Result<Self> {
        let vocab = directory.join("vocab.json");
        let merges = directory.join("merges.txt");
        let bpe = BPE::from_file(
            vocab.to_str().context("Invalid model path")?,
            merges.to_str().context("Invalid model path")?,
        )
        .build()
        .map_err(tokenizer_error)?;
        let mut tokenizer = Tokenizer::new(bpe);
        let split = Split::new(SplitPattern::Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+".into()), SplitDelimiterBehavior::Isolated, false).map_err(tokenizer_error)?;
        tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
            split.into(),
            ByteLevel::new(false, false, false).into(),
        ])));
        tokenizer.with_decoder(Some(ByteLevel::default()));
        let config: serde_json::Value = serde_json::from_reader(std::fs::File::open(
            directory.join("tokenizer_config.json"),
        )?)?;
        let mut special: Vec<(u32, String)> = config["added_tokens_decoder"]
            .as_object()
            .context("Missing special tokens")?
            .iter()
            .map(|(id, value)| {
                Ok((
                    id.parse()?,
                    value["content"]
                        .as_str()
                        .context("Missing token content")?
                        .to_owned(),
                ))
            })
            .collect::<Result<_>>()?;
        special.sort_by_key(|(id, _)| *id);
        for (id, content) in special {
            tokenizer.add_special_tokens(&[AddedToken::from(content.clone(), true)]);
            ensure!(
                tokenizer.token_to_id(&content) == Some(id),
                "Special token ID mismatch: {content}"
            );
        }
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(tokenizer_error)?
            .get_ids()
            .to_vec())
    }

    pub fn suffix(&self, language: Language) -> Result<Vec<u32>> {
        let mut tokens = SUFFIX.to_vec();
        if language != Language::Auto {
            let name = if language == Language::Chinese {
                "Chinese"
            } else {
                "English"
            };
            tokens.extend(self.encode(&format!("language {name}"))?);
            tokens.push(ASR_TEXT);
        }
        Ok(tokens)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens, true).map_err(tokenizer_error)
    }
}
