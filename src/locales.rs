use serde::{Deserialize, Serialize};
use serde_yaml::from_str;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::UNIX_EPOCH;
use thiserror::Error;

#[derive(Error, Debug, Serialize, Deserialize)]
pub enum LocaleError {
    #[error("Failed to load locale file: {0}")]
    LoadError(String),
    #[error("Failed to parse locale file: {0}")]
    ParseError(String),
    #[error("Locale not found: {0}")]
    LocaleNotFound(String),
}

impl From<std::io::Error> for LocaleError {
    fn from(err: std::io::Error) -> Self {
        LocaleError::LoadError(err.to_string())
    }
}

impl From<serde_yaml::Error> for LocaleError {
    fn from(err: serde_yaml::Error) -> Self {
        LocaleError::ParseError(err.to_string())
    }
}

#[derive(Clone)]
pub struct Locales {
    translations: Arc<RwLock<HashMap<String, HashMap<String, String>>>>,
    default_locale: Arc<RwLock<String>>,
    file_timestamps: Arc<RwLock<HashMap<String, u64>>>,
    base_path: String,
}

impl Locales {
    fn get_file_timestamp(path: &Path) -> Result<u64, std::io::Error> {
        let metadata = std::fs::metadata(path)?;
        let modified = metadata.modified()?;
        Ok(modified.duration_since(UNIX_EPOCH).unwrap().as_secs())
    }

    fn flatten_yaml(
        value: &serde_yaml::Value,
        result: &mut HashMap<String, String>,
        prefix: String,
    ) {
        match value {
            serde_yaml::Value::Mapping(map) => {
                for (k, v) in map {
                    if let Some(key) = k.as_str() {
                        let new_prefix = if prefix.is_empty() {
                            key.to_string()
                        } else {
                            format!("{}.{}", prefix, key)
                        };
                        Self::flatten_yaml(v, result, new_prefix);
                    }
                }
            }
            serde_yaml::Value::String(s) => {
                result.insert(prefix, s.clone());
            }
            _ => {}
        }
    }

    fn load_locale_file(&self, path: &Path) -> Result<(), LocaleError> {
        let locale = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| LocaleError::LoadError(path.display().to_string()))?
            .to_string();

        let content = std::fs::read_to_string(path).map_err(LocaleError::from)?;
        let value: serde_yaml::Value = from_str(&content).map_err(|e| {
            LocaleError::ParseError(format!("Failed to parse {}: {}", path.display(), e))
        })?;

        let mut string_data = HashMap::new();
        if let serde_yaml::Value::Mapping(map) = value {
            Self::flatten_yaml(&serde_yaml::Value::Mapping(map), &mut string_data, String::new());
        }

        let mut translations = self.translations.write().unwrap();
        translations.insert(locale.clone(), string_data);

        let mut timestamps = self.file_timestamps.write().unwrap();
        timestamps.insert(locale, Self::get_file_timestamp(path)?);

        Ok(())
    }

    fn check_and_reload(&self, locale: &str) -> Result<(), LocaleError> {
        let path = Path::new(&self.base_path).join(format!("{}.yml", locale));
        if !path.exists() {
            return Err(LocaleError::LocaleNotFound(locale.to_string()));
        }

        let current_timestamp = Self::get_file_timestamp(&path)?;
        let timestamps = self.file_timestamps.read().unwrap();
        if let Some(last_timestamp) = timestamps.get(locale) {
            if *last_timestamp < current_timestamp {
                drop(timestamps);
                self.load_locale_file(&path)?;
            }
        } else {
            drop(timestamps);
            self.load_locale_file(&path)?;
        }

        Ok(())
    }

    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, LocaleError> {
        let base_path = path
            .as_ref()
            .to_str()
            .ok_or_else(|| LocaleError::LoadError("Invalid path".to_string()))?
            .to_string();

        let locales = Self {
            translations: Arc::new(RwLock::new(HashMap::new())),
            default_locale: Arc::new(RwLock::new(String::new())),
            file_timestamps: Arc::new(RwLock::new(HashMap::new())),
            base_path,
        };

        let entries = fs::read_dir(path).map_err(LocaleError::from)?;
        for entry in entries {
            let entry = entry.map_err(LocaleError::from)?;
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "yml") {
                locales.load_locale_file(&path)?;
            }
        }

        Ok(locales)
    }

    pub fn set_default(&mut self, locale: &str) -> Result<(), LocaleError> {
        if self.translations.read().unwrap().contains_key(locale) {
            *self.default_locale.write().unwrap() = locale.to_string();
            Ok(())
        } else {
            Err(LocaleError::LocaleNotFound(locale.to_string()))
        }
    }

    pub fn t(&self, key: &str) -> String {
        self.translate(&self.default_locale.read().unwrap(), key)
            .unwrap_or_else(|_| key.to_string())
    }

    pub fn translate(&self, locale: &str, key: &str) -> Result<String, LocaleError> {
        self.check_and_reload(locale)?;

        let translations = self.translations.read().unwrap();
        translations
            .get(locale)
            .and_then(|data| data.get(key).map(|s| s.to_string()))
            .ok_or_else(|| LocaleError::LocaleNotFound(key.to_string()))
    }
}
