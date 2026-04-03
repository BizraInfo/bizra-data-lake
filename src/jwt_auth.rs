// src/jwt_auth.rs - JWT Authentication Middleware with Token Rotation
// Standing on Shoulders of Giants Protocol: OAuth 2.0 / JWT (RFC 7519) standards
// Extends BIZRA Ihsān security dimensions (safety: 0.22, correctness: 0.22)

use crate::errors::BridgeError;
use chrono::{Duration, Utc};
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

const JWT_ALGORITHM: &str = "RS256";
const TOKEN_EXPIRY_SECONDS: i64 = 3600;
const REFRESH_EXPIRY_SECONDS: i64 = 86400;
const MAX_TOKEN_PER_USER: usize = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtClaims {
    pub sub: String,
    pub iat: i64,
    pub exp: i64,
    pub roles: Vec<String>,
    pub scopes: Vec<String>,
    pub token_id: String,
    pub refresh_id: Option<String>,
}

#[derive(Clone)]
pub struct JwtAuthenticator {
    signing_key: Arc<SigningKey>,
    verifying_key: Arc<VerifyingKey>,
    token_store: Arc<RwLock<HashMap<String, TokenEntry>>>,
    refresh_store: Arc<RwLock<HashMap<String, RefreshEntry>>>,
    revocation_list: Arc<RwLock<HashMap<String, i64>>>,
}

struct TokenEntry {
    user_id: String,
    expires_at: i64,
    roles: Vec<String>,
    scopes: Vec<String>,
    token_id: String,
}

struct RefreshEntry {
    user_id: String,
    expires_at: i64,
    rotation_count: u32,
}

#[derive(Debug, Serialize)]
pub struct JwtToken {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub token_type: String,
    pub expires_in: u64,
    pub roles: Vec<String>,
    pub scopes: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct TokenValidationResult {
    pub valid: bool,
    pub user_id: String,
    pub roles: Vec<String>,
    pub scopes: Vec<String>,
    pub expires_at: i64,
    pub token_id: String,
    pub error: Option<String>,
}

impl JwtAuthenticator {
    pub fn generate_keypair() -> (SigningKey, VerifyingKey) {
        use rand::rngs::OsRng;
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        (signing_key, verifying_key)
    }

    pub fn new(signing_key: SigningKey) -> Self {
        Self {
            signing_key: Arc::new(signing_key),
            verifying_key: Arc::new(signing_key.verifying_key()),
            token_store: Arc::new(RwLock::new(HashMap::new())),
            refresh_store: Arc::new(RwLock::new(HashMap::new())),
            revocation_list: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn verifying_key_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }

    pub async fn issue_token(
        &self,
        user_id: String,
        roles: Vec<String>,
        scopes: Vec<String>,
    ) -> Result<JwtToken, BridgeError> {
        let now = Utc::now().timestamp();
        let token_id = uuid::Uuid::new_v4().to_string();
        
        let claims = JwtClaims {
            sub: user_id.clone(),
            iat: now,
            exp: now + TOKEN_EXPIRY_SECONDS,
            roles: roles.clone(),
            scopes: scopes.clone(),
            token_id: token_id.clone(),
            refresh_id: None,
        };

        let header = format!("{{\"alg\":\"{}\",\"typ\":\"JWT\"}}", JWT_ALGORITHM);
        let header_b64 = base64_url_encode(header.as_bytes());
        let payload_b64 = base64_url_encode(serde_json::to_vec(&claims)?);
        let message = format!("{}.{}", header_b64, payload_b64);
        let signature = self.signing_key.sign(message.as_bytes());
        let signature_b64 = base64_url_encode(signature.to_bytes());
        let access_token = format!("{}.{}", message, signature_b64);

        let refresh_id = uuid::Uuid::new_v4().to_string();
        let refresh_claims = JwtClaims {
            sub: user_id.clone(),
            iat: now,
            exp: now + REFRESH_EXPIRY_SECONDS,
            roles: roles.clone(),
            scopes: scopes.clone(),
            token_id: token_id.clone(),
            refresh_id: Some(refresh_id.clone()),
        };

        let refresh_header = format!("{{\"alg\":\"{}\",\"typ\":\"JWT\"}}", JWT_ALGORITHM);
        let refresh_header_b64 = base64_url_encode(refresh_header.as_bytes());
        let refresh_payload_b64 = base64_url_encode(serde_json::to_vec(&refresh_claims)?);
        let refresh_message = format!("{}.{}", refresh_header_b64, refresh_payload_b64);
        let refresh_signature = self.signing_key.sign(refresh_message.as_bytes());
        let refresh_signature_b64 = base64_url_encode(refresh_signature.to_bytes());
        let refresh_token = format!("{}.{}", refresh_message, refresh_signature_b64);

        let entry = TokenEntry {
            user_id: user_id.clone(),
            expires_at: claims.exp,
            roles: roles.clone(),
            scopes: scopes.clone(),
            token_id: token_id.clone(),
        };
        
        {
            let mut store = self.token_store.write().await;
            let user_tokens = store.entry(user_id.clone()).or_insert_with(Vec::new);
            while user_tokens.len() >= MAX_TOKEN_PER_USER {
                user_tokens.remove(0);
            }
            user_tokens.push(entry);
        }

        let refresh_entry = RefreshEntry {
            user_id: user_id.clone(),
            expires_at: refresh_claims.exp,
            rotation_count: 0,
        };
        {
            let mut refresh = self.refresh_store.write().await;
            refresh.insert(refresh_id.clone(), refresh_entry);
        }

        Ok(JwtToken {
            access_token,
            refresh_token: Some(refresh_token),
            token_type: "Bearer".to_string(),
            expires_in: TOKEN_EXPIRY_SECONDS as u64,
            roles,
            scopes,
        })
    }

    pub async fn validate_token(&self, token: &str) -> TokenValidationResult {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return TokenValidationResult {
                valid: false,
                user_id: String::new(),
                roles: Vec::new(),
                scopes: Vec::new(),
                expires_at: 0,
                token_id: String::new(),
                error: Some("Invalid token format".to_string()),
            };
        }

        let message = format!("{}.{}", parts[0], parts[1]);
        if let Ok(signature_bytes) = base64_url_decode(parts[2]) {
            if let Ok(signature) = ed25519_dalek::Signature::from_slice(&signature_bytes) {
                if self.verifying_key.verify(message.as_bytes(), &signature).is_err() {
                    return TokenValidationResult {
                        valid: false,
                        user_id: String::new(),
                        roles: Vec::new(),
                        scopes: Vec::new(),
                        expires_at: 0,
                        token_id: String::new(),
                        error: Some("Signature verification failed".to_string()),
                    };
                }
            }
        }

        let payload = match base64_url_decode(parts[1]) {
            Ok(p) => p,
            Err(_) => return TokenValidationResult {
                valid: false,
                user_id: String::new(),
                roles: Vec::new(),
                scopes: Vec::new(),
                expires_at: 0,
                token_id: String::new(),
                error: Some("Invalid payload encoding".to_string()),
            },
        };

        let claims: JwtClaims = match serde_json::from_slice(&payload) {
            Ok(c) => c,
            Err(_) => return TokenValidationResult {
                valid: false,
                user_id: String::new(),
                roles: Vec::new(),
                scopes: Vec::new(),
                expires_at: 0,
                token_id: String::new(),
                error: Some("Invalid claims JSON".to_string()),
            },
        };

        let now = Utc::now().timestamp();
        if claims.exp < now {
            return TokenValidationResult {
                valid: false,
                user_id: claims.sub,
                roles: claims.roles,
                scopes: claims.scopes,
                expires_at: claims.exp,
                token_id: claims.token_id,
                error: Some("Token expired".to_string()),
            };
        }

        {
            let revoked = self.revocation_list.read().await;
            if let Some(revoked_at) = revoked.get(&claims.token_id) {
                if *revoked_at > claims.iat {
                    return TokenValidationResult {
                        valid: false,
                        user_id: claims.sub,
                        roles: claims.roles,
                        scopes: claims.scopes,
                        expires_at: claims.exp,
                        token_id: claims.token_id,
                        error: Some("Token revoked".to_string()),
                    };
                }
            }
        }

        TokenValidationResult {
            valid: true,
            user_id: claims.sub,
            roles: claims.roles,
            scopes: claims.scopes,
            expires_at: claims.exp,
            token_id: claims.token_id,
            error: None,
        }
    }

    pub async fn refresh_access_token(&self, refresh_token: &str) -> Result<JwtToken, BridgeError> {
        let parts: Vec<&str> = refresh_token.split('.').collect();
        if parts.len() != 3 {
            return Err(BridgeError::Auth("Invalid refresh token format".to_string()));
        }

        let payload = base64_url_decode(parts[1])
            .map_err(|e| BridgeError::Auth(format!("Payload decode error: {}", e)))?;
        let claims: JwtClaims = serde_json::from_slice(&payload)
            .map_err(|e| BridgeError::Auth(format!("Claims parse error: {}", e)))?;

        let refresh_id = claims.refresh_id.clone()
            .ok_or_else(|| BridgeError::Auth("Not a refresh token".to_string()))?;

        self.revoke_token(&claims.token_id).await?;

        self.issue_token(claims.sub, claims.roles, claims.scopes).await
    }

    pub async fn revoke_token(&self, token_id: &str) -> Result<(), BridgeError> {
        let now = Utc::now().timestamp();
        {
            let mut revoked = self.revocation_list.write().await;
            revoked.insert(token_id.to_string(), now);
        }
        
        {
            let mut store = self.token_store.write().await;
            for (_user_id, tokens) in store.iter_mut() {
                tokens.retain(|t| t.token_id != token_id);
            }
        }

        Ok(())
    }

    pub async fn revoke_all_user_tokens(&self, user_id: &str) -> Result<u32, BridgeError> {
        let mut count = 0;
        {
            let mut store = self.token_store.write().await;
            if let Some(tokens) = store.get_mut(user_id) {
                let now = Utc::now().timestamp();
                for token in tokens.iter() {
                    let mut revoked = self.revocation_list.write().await;
                    revoked.insert(token.token_id.clone(), now);
                    count += 1;
                }
                tokens.clear();
            }
        }
        
        Ok(count)
    }

    pub async fn get_user_sessions(&self, user_id: &str) -> Vec<TokenValidationResult> {
        let store = self.token_store.read().await;
        let user_tokens = store.get(user_id);
        
        match user_tokens {
            Some(tokens) => tokens.iter().map(|t| TokenValidationResult {
                valid: t.expires_at > Utc::now().timestamp(),
                user_id: t.user_id.clone(),
                roles: t.roles.clone(),
                scopes: t.scopes.clone(),
                expires_at: t.expires_at,
                token_id: t.token_id.clone(),
                error: None,
            }).collect(),
            None => Vec::new(),
        }
    }
}

fn base64_url_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        result.push(ALPHABET[(b0 >> 2) as usize] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4 | b1 >> 4) as usize] as char);
        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2 | b2 >> 6) as usize] as char);
        }
        if chunk.len() > 2 {
            result.push(ALPHABET[(b2 & 0x3f) as usize] as char);
        }
    }
    result
}

fn base64_url_decode(input: &str) -> Result<Vec<u8>, ()> {
    const DECODE: [i8; 256] = {
        let mut arr = [-1i8; 256];
        arr[b'A' as usize] = 0; arr[b'B' as usize] = 1; arr[b'C' as usize] = 2;
        arr[b'D' as usize] = 3; arr[b'E' as usize] = 4; arr[b'F' as usize] = 5;
        arr[b'G' as usize] = 6; arr[b'H' as usize] = 7; arr[b'I' as usize] = 8;
        arr[b'J' as usize] = 9; arr[b'K' as usize] = 10; arr[b'L' as usize] = 11;
        arr[b'M' as usize] = 12; arr[b'N' as usize] = 13; arr[b'O' as usize] = 14;
        arr[b'P' as usize] = 15; arr[b'Q' as usize] = 16; arr[b'R' as usize] = 17;
        arr[b'S' as usize] = 18; arr[b'T' as usize] = 19; arr[b'U' as usize] = 20;
        arr[b'V' as usize] = 21; arr[b'W' as usize] = 22; arr[b'X' as usize] = 23;
        arr[b'Y' as usize] = 24; arr[b'Z' as usize] = 25; arr[b'a' as usize] = 26;
        arr[b'b' as usize] = 27; arr[b'c' as usize] = 28; arr[b'd' as usize] = 29;
        arr[b'e' as usize] = 30; arr[b'f' as usize] = 31; arr[b'g' as usize] = 32;
        arr[b'h' as usize] = 33; arr[b'i' as usize] = 34; arr[b'j' as usize] = 35;
        arr[b'k' as usize] = 36; arr[b'l' as usize] = 37; arr[b'm' as usize] = 38;
        arr[b'n' as usize] = 39; arr[b'o' as usize] = 40; arr[b'p' as usize] = 41;
        arr[b'q' as usize] = 42; arr[b'r' as usize] = 43; arr[b's' as usize] = 44;
        arr[b't' as usize] = 45; arr[b'u' as usize] = 46; arr[b'v' as usize] = 47;
        arr[b'w' as usize] = 48; arr[b'x' as usize] = 49; arr[b'y' as usize] = 50;
        arr[b'z' as usize] = 51; arr[b'0' as usize] = 52; arr[b'1' as usize] = 53;
        arr[b'2' as usize] = 54; arr[b'3' as usize] = 55; arr[b'4' as usize] = 56;
        arr[b'5' as usize] = 57; arr[b'6' as usize] = 58; arr[b'7' as usize] = 59;
        arr[b'8' as usize] = 60; arr[b'9' as usize] = 61; arr[b'-' as usize] = 62;
        arr[b'_' as usize] = 63;
        arr
    };
    
    let mut result = Vec::new();
    let chars: Vec<u8> = input.bytes().filter(|&b| b != b'=').collect();
    for chunk in chars.chunks(4) {
        let mut buf = [0u8; 4];
        for (i, &c) in chunk.iter().enumerate() {
            let val = DECODE[c as usize];
            if val < 0 { return Err(()); }
            buf[i] = val as u8;
        }
        result.push((buf[0] << 2 | buf[1] >> 4) as u8);
        if chunk.len() > 2 {
            result.push((buf[1] << 4 | buf[2] >> 2) as u8);
        }
        if chunk.len() > 3 {
            result.push((buf[2] << 6 | buf[3]) as u8);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_issue_and_validate() {
        let (signing_key, _) = JwtAuthenticator::generate_keypair();
        let auth = JwtAuthenticator::new(signing_key);

        let token = auth.issue_token(
            "user_001".to_string(),
            vec!["pat".to_string()],
            vec!["execute".to_string(), "read".to_string()],
        ).await.unwrap();

        let validation = auth.validate_token(&token.access_token).await;
        assert!(validation.valid);
        assert_eq!(validation.user_id, "user_001");
    }
}