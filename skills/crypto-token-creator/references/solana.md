# Solana Token Development Reference

## Toolchain Setup

### Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Solana CLI
sh -c "$(curl -sSfL https://release.anza.xyz/stable/install)"

# Install Anchor (framework for Solana programs)
cargo install --git https://github.com/coral-xyz/anchor avm --force
avm install latest
avm use latest

# Install Node.js dependencies for testing
npm install -g yarn
```

### Project Initialization
```bash
anchor init my-token
cd my-token
```

## Project Structure
```
my-token/
├── programs/
│   └── my-token/
│       ├── src/
│       │   └── lib.rs
│       └── Cargo.toml
├── tests/
│   └── my-token.ts
├── migrations/
│   └── deploy.ts
├── Anchor.toml
├── Cargo.toml
├── package.json
└── tsconfig.json
```
---

## SPL Token Creation (No Program Needed)

For standard fungible tokens on Solana, you don't need to write a custom program.
The SPL Token program handles everything. Use the CLI or a script:

### CLI Method (Simplest)
```bash
# Create token mint
spl-token create-token --decimals 9

# Create token account
spl-token create-account <MINT_ADDRESS>

# Mint tokens
spl-token mint <MINT_ADDRESS> 1000000000

# Disable future minting (fixed supply)
spl-token authorize <MINT_ADDRESS> mint --disable
```

### Script Method (TypeScript)
```typescript
import {
  Connection, Keypair, clusterApiUrl
} from "@solana/web3.js";
import {
  createMint, getOrCreateAssociatedTokenAccount, mintTo,
  setAuthority, AuthorityType
} from "@solana/spl-token";
async function createToken() {
  const connection = new Connection(clusterApiUrl("devnet"), "confirmed");
  const payer = Keypair.fromSecretKey(/* load from file */);

  // Create mint
  const mint = await createMint(
    connection,
    payer,
    payer.publicKey,    // mint authority
    payer.publicKey,    // freeze authority (null to disable)
    9                   // decimals
  );
  console.log("Mint address:", mint.toBase58());

  // Create token account
  const tokenAccount = await getOrCreateAssociatedTokenAccount(
    connection, payer, mint, payer.publicKey
  );

  // Mint initial supply
  await mintTo(
    connection, payer, mint,
    tokenAccount.address,
    payer,              // mint authority
    1_000_000_000n * 10n ** 9n  // amount with decimals
  );

  // Disable minting (makes supply fixed)
  await setAuthority(
    connection, payer, mint,
    payer,              // current authority
    AuthorityType.MintTokens,
    null                // new authority (null = disable)
  );

  console.log("Token created successfully!");
  console.log("Mint:", mint.toBase58());
  console.log("Token Account:", tokenAccount.address.toBase58());
}

createToken();
```

---

## Token-2022 (Advanced SPL Tokens)
Token-2022 is the next-generation SPL token program with built-in extensions:

### Available Extensions:
- **Transfer fees** — automatic fee on every transfer (like ERC-20 tax)
- **Interest-bearing** — token balance grows over time
- **Non-transferable** — soulbound tokens
- **Confidential transfers** — encrypted transfer amounts
- **Transfer hook** — custom logic on every transfer
- **Metadata** — on-chain metadata without Metaplex
- **Permanent delegate** — authority that can transfer/burn any holder's tokens

### Token-2022 with Transfer Fee
```typescript
import {
  ExtensionType, createInitializeMintInstruction,
  createInitializeTransferFeeConfigInstruction,
  getMintLen, TOKEN_2022_PROGRAM_ID
} from "@solana/spl-token";
import {
  Connection, Keypair, SystemProgram, Transaction,
  sendAndConfirmTransaction
} from "@solana/web3.js";

async function createFeeToken() {
  const connection = new Connection(clusterApiUrl("devnet"), "confirmed");
  const payer = Keypair.fromSecretKey(/* load from file */);
  const mintKeypair = Keypair.generate();

  const extensions = [ExtensionType.TransferFeeConfig];
  const mintLen = getMintLen(extensions);
  const lamports = await connection.getMinimumBalanceForRentExemption(mintLen);
  const transaction = new Transaction().add(
    SystemProgram.createAccount({
      fromPubkey: payer.publicKey,
      newAccountPubkey: mintKeypair.publicKey,
      space: mintLen,
      lamports,
      programId: TOKEN_2022_PROGRAM_ID,
    }),
    createInitializeTransferFeeConfigInstruction(
      mintKeypair.publicKey,
      payer.publicKey,       // fee config authority
      payer.publicKey,       // withdraw withheld authority
      250,                   // 2.5% fee (basis points)
      BigInt(5_000_000_000), // max fee (5 tokens)
      TOKEN_2022_PROGRAM_ID
    ),
    createInitializeMintInstruction(
      mintKeypair.publicKey,
      9,                     // decimals
      payer.publicKey,       // mint authority
      null,                  // freeze authority
      TOKEN_2022_PROGRAM_ID
    )
  );

  await sendAndConfirmTransaction(connection, transaction, [payer, mintKeypair]);
  console.log("Fee token created:", mintKeypair.publicKey.toBase58());
}
```

---

## Custom Anchor Program (Advanced Use Cases)

For custom token logic beyond what SPL provides:

### Token with Custom Mint Logic
```rust
use anchor_lang::prelude::*;
use anchor_spl::token::{self, Mint, Token, TokenAccount, MintTo};
declare_id!("YOUR_PROGRAM_ID_HERE");

#[program]
pub mod my_token {
    use super::*;

    pub fn initialize(
        ctx: Context<Initialize>,
        max_supply: u64,
        decimals: u8,
    ) -> Result<()> {
        let config = &mut ctx.accounts.config;
        config.authority = ctx.accounts.authority.key();
        config.mint = ctx.accounts.mint.key();
        config.max_supply = max_supply;
        config.total_minted = 0;
        config.decimals = decimals;
        Ok(())
    }

    pub fn mint_tokens(ctx: Context<MintTokens>, amount: u64) -> Result<()> {
        let config = &mut ctx.accounts.config;
        require!(
            config.total_minted + amount <= config.max_supply,
            ErrorCode::ExceedsMaxSupply
        );

        config.total_minted += amount;

        let seeds = &[b"config".as_ref(), &[ctx.bumps.config]];
        let signer_seeds = &[&seeds[..]];

        token::mint_to(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                MintTo {
                    mint: ctx.accounts.mint.to_account_info(),
                    to: ctx.accounts.token_account.to_account_info(),
                    authority: ctx.accounts.config.to_account_info(),
                },
                signer_seeds,
            ),
            amount,
        )?;
        Ok(())
    }
}
#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + Config::INIT_SPACE,
        seeds = [b"config"],
        bump
    )]
    pub config: Account<'info, Config>,
    #[account(mut)]
    pub mint: Account<'info, Mint>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct MintTokens<'info> {
    #[account(
        mut,
        seeds = [b"config"],
        bump,
        has_one = authority,
    )]
    pub config: Account<'info, Config>,
    #[account(mut)]
    pub mint: Account<'info, Mint>,
    #[account(mut)]
    pub token_account: Account<'info, TokenAccount>,
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[account]
#[derive(InitSpace)]
pub struct Config {
    pub authority: Pubkey,
    pub mint: Pubkey,
    pub max_supply: u64,
    pub total_minted: u64,
    pub decimals: u8,
}
#[error_code]
pub enum ErrorCode {
    #[msg("Exceeds maximum supply")]
    ExceedsMaxSupply,
}
```

---

## Metaplex NFT Creation

### Using Metaplex JS SDK
```typescript
import { Metaplex, keypairIdentity, bundlrStorage } from "@metaplex-foundation/js";
import { Connection, Keypair, clusterApiUrl } from "@solana/web3.js";

async function createNFTCollection() {
  const connection = new Connection(clusterApiUrl("devnet"));
  const wallet = Keypair.fromSecretKey(/* load from file */);

  const metaplex = Metaplex.make(connection)
    .use(keypairIdentity(wallet))
    .use(bundlrStorage({ address: "https://devnet.bundlr.network" }));

  // Create collection NFT
  const { nft: collection } = await metaplex.nfts().create({
    name: "My Collection",
    symbol: "MYCOL",
    uri: "https://arweave.net/collection-metadata.json",
    sellerFeeBasisPoints: 500, // 5% royalty
    isCollection: true,
  });

  // Mint NFT into collection
  const { nft } = await metaplex.nfts().create({
    name: "My NFT #1",
    symbol: "MYCOL",
    uri: "https://arweave.net/nft-1-metadata.json",
    sellerFeeBasisPoints: 500,
    collection: collection.address,
  });

  // Verify collection
  await metaplex.nfts().verifyCollection({
    mintAddress: nft.address,
    collectionMintAddress: collection.address,
  });

  console.log("Collection:", collection.address.toBase58());
  console.log("NFT:", nft.address.toBase58());
}
```
### NFT Metadata Format (Metaplex Standard)
```json
{
  "name": "My NFT #1",
  "symbol": "MYCOL",
  "description": "A unique digital collectible",
  "image": "https://arweave.net/image.png",
  "animation_url": "",
  "external_url": "https://myproject.com",
  "attributes": [
    { "trait_type": "Background", "value": "Blue" },
    { "trait_type": "Rarity", "value": "Legendary" }
  ],
  "properties": {
    "files": [
      { "uri": "https://arweave.net/image.png", "type": "image/png" }
    ],
    "category": "image",
    "creators": [
      { "address": "CREATOR_PUBKEY", "share": 100 }
    ]
  }
}
```

---

## Deployment

### Anchor.toml Configuration
```toml
[features]
seeds = false
skip-lint = false

[programs.devnet]
my_token = "YOUR_PROGRAM_ID"

[registry]
url = "https://api.apr.dev"

[provider]
cluster = "devnet"
wallet = "~/.config/solana/id.json"

[scripts]
test = "yarn run ts-mocha -p ./tsconfig.json -t 1000000 tests/**/*.ts"
```
### Deploy Commands
```bash
# Build
anchor build

# Get program ID
solana address -k target/deploy/my_token-keypair.json

# Update program ID in lib.rs and Anchor.toml

# Deploy to devnet
anchor deploy --provider.cluster devnet

# Deploy to mainnet
anchor deploy --provider.cluster mainnet-beta
```

---

## Cost Estimates

| Operation | ~Cost (SOL) | ~Cost (USD at $150/SOL) |
|-----------|-------------|------------------------|
| SPL Token creation | 0.002 | ~$0.30 |
| Token-2022 creation | 0.003 | ~$0.45 |
| Anchor program deploy | 2-5 | ~$300-750 |
| NFT mint (Metaplex) | 0.01 | ~$1.50 |
| Collection creation | 0.02 | ~$3.00 |

## Testnet Faucet
- **Solana Devnet**: `solana airdrop 2` (CLI) or https://faucet.solana.com