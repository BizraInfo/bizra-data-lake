# TON (The Open Network) Token Development Reference

## Toolchain Setup

### Prerequisites
```bash
# Install Node.js (18+)
# Install TON development tools
npm install -g @ton/blueprint

# Create new project
npm create ton@latest my-token
# Select: Tact language, empty contract template
cd my-token
npm install
```

### Alternative: FunC (lower-level)
```bash
# FunC compiler comes with blueprint
# Use Tact for new projects — it compiles to FunC and is much more readable
```

## Project Structure (Blueprint)
```
my-token/
├── contracts/
│   └── my_jetton.tact
├── scripts/
│   └── deployMyJetton.ts
├── tests/
│   └── MyJetton.spec.ts
├── wrappers/
│   └── MyJetton.ts
├── blueprint.config.ts
├── package.json
└── tsconfig.json
```

---

## Jetton (Fungible Token) — TEP-74 Standard

TON's fungible token standard is called "Jetton" (TEP-74). Unlike Ethereum where
one contract tracks all balances, TON uses a master-wallet architecture:
- **Jetton Master** — stores metadata, total supply, minting logic
- **Jetton Wallet** — one per holder, stores that user's balance

This is important to understand because it means each user's tokens live in their
own smart contract (wallet), not in a central mapping.
### Jetton in Tact
```tact
import "@stdlib/deploy";
import "@stdlib/ownable";
import "@stdlib/jetton";

contract MyJetton with Jetton, Deployable, Ownable {
    totalSupply: Int as coins;
    owner: Address;
    mintable: Bool;
    content: Cell;
    max_supply: Int as coins;

    init(owner: Address, content: Cell, max_supply: Int) {
        self.totalSupply = 0;
        self.owner = owner;
        self.mintable = true;
        self.content = content;
        self.max_supply = max_supply;
    }

    receive(msg: JettonMint) {
        self.requireOwner();
        require(self.mintable, "Minting disabled");
        require(
            self.totalSupply + msg.amount <= self.max_supply,
            "Exceeds max supply"
        );
        self.mint(msg.receiver, msg.amount, msg.receiver);
    }

    receive(msg: DisableMint) {
        self.requireOwner();
        self.mintable = false;
    }

    override inline fun calculate_jetton_wallet_init(owner: Address): StateInit {
        return initOf MyJettonWallet(owner, myAddress());
    }
}

contract MyJettonWallet with JettonWallet {
    balance: Int as coins = 0;
    owner: Address;
    jetton_master: Address;

    init(owner: Address, jetton_master: Address) {
        self.owner = owner;
        self.jetton_master = jetton_master;
    }

    override inline fun calculate_jetton_wallet_init(owner: Address): StateInit {
        return initOf MyJettonWallet(owner, self.jetton_master);
    }
}

message JettonMint {
    receiver: Address;
    amount: Int as coins;
}

message DisableMint {}
```

### Jetton Metadata (On-chain or Off-chain)

**Off-chain metadata (recommended):**
```json
{
  "name": "My Token",
  "symbol": "MTK",
  "description": "A utility token on TON",
  "decimals": "9",
  "image": "https://example.com/logo.png"
}
```

Host the JSON file and encode the URL as a Cell:
```typescript
import { beginCell } from "@ton/core";

function jettonContentToCell(uri: string): Cell {
    return beginCell()
        .storeUint(0x01, 8) // off-chain flag
        .storeStringTail(uri)
        .endCell();
}
```

---

## NFT Collection — TEP-62 Standard

Like Jettons, TON NFTs use a collection + item architecture:
- **NFT Collection** — stores metadata, manages item deployment
- **NFT Item** — one contract per NFT

### NFT Collection in Tact
```tact
import "@stdlib/deploy";
import "@stdlib/ownable";

contract NftCollection with Deployable, Ownable {
    owner: Address;
    nextItemIndex: Int as uint64;
    collectionContent: Cell;
    royaltyParams: RoyaltyParams;
    maxSupply: Int as uint64;

    init(
        owner: Address,
        collectionContent: Cell,
        royaltyParams: RoyaltyParams,
        maxSupply: Int
    ) {
        self.owner = owner;
        self.nextItemIndex = 0;
        self.collectionContent = collectionContent;
        self.royaltyParams = royaltyParams;
        self.maxSupply = maxSupply;
    }

    receive(msg: MintNft) {
        self.requireOwner();
        require(self.nextItemIndex < self.maxSupply, "Max supply reached");

        let itemInit = initOf NftItem(
            myAddress(),
            self.nextItemIndex,
            msg.owner,
            msg.content
        );

        send(SendParameters{
            to: contractAddress(itemInit),
            value: ton("0.05"),
            mode: 0,
            body: NftItemDeploy{
                owner: msg.owner,
                content: msg.content,
                index: self.nextItemIndex
            }.toCell(),
            code: itemInit.code,
            data: itemInit.data
        });

        self.nextItemIndex += 1;
    }

    get fun get_collection_data(): CollectionData {
        return CollectionData{
            nextItemIndex: self.nextItemIndex,
            collectionContent: self.collectionContent,
            ownerAddress: self.owner
        };
    }
}

contract NftItem {
    collection: Address;
    index: Int as uint64;
    owner: Address;
    content: Cell;
    initialized: Bool;

    init(collection: Address, index: Int, owner: Address, content: Cell) {
        self.collection = collection;
        self.index = index;
        self.owner = owner;
        self.content = content;
        self.initialized = false;
    }

    receive(msg: NftItemDeploy) {
        require(!self.initialized, "Already initialized");
        require(sender() == self.collection, "Only collection");
        self.owner = msg.owner;
        self.content = msg.content;
        self.initialized = true;
    }

    receive(msg: NftTransfer) {
        require(sender() == self.owner, "Not owner");
        self.owner = msg.newOwner;
        // Send excess to response destination
        send(SendParameters{
            to: msg.responseDestination,
            value: 0,
            mode: SendRemainingValue,
            body: NftOwnershipAssigned{
                previousOwner: sender(),
                forwardPayload: msg.forwardPayload
            }.toCell()
        });
    }

    get fun get_nft_data(): NftData {
        return NftData{
            initialized: self.initialized,
            index: self.index,
            collectionAddress: self.collection,
            ownerAddress: self.owner,
            content: self.content
        };
    }
}

// Messages
message MintNft {
    owner: Address;
    content: Cell;
}

message NftItemDeploy {
    owner: Address;
    content: Cell;
    index: Int as uint64;
}

message NftTransfer {
    newOwner: Address;
    responseDestination: Address;
    forwardPayload: Cell?;
}

message NftOwnershipAssigned {
    previousOwner: Address;
    forwardPayload: Cell?;
}

// Data structures
struct RoyaltyParams {
    numerator: Int as uint16;   // e.g., 5
    denominator: Int as uint16; // e.g., 100  → 5% royalty
    destination: Address;
}

struct CollectionData {
    nextItemIndex: Int;
    collectionContent: Cell;
    ownerAddress: Address;
}

struct NftData {
    initialized: Bool;
    index: Int;
    collectionAddress: Address;
    ownerAddress: Address;
    content: Cell;
}
```

---

## Deployment Scripts

### Deploy Jetton (Blueprint)
```typescript
import { toNano } from "@ton/core";
import { MyJetton } from "../wrappers/MyJetton";
import { NetworkProvider } from "@ton/blueprint";

export async function run(provider: NetworkProvider) {
    const owner = provider.sender().address!;

    const content = beginCell()
        .storeUint(0x01, 8)
        .storeStringTail("https://example.com/jetton-metadata.json")
        .endCell();

    const jetton = provider.open(
        await MyJetton.fromInit(
            owner,
            content,
            toNano("1000000000") // max supply: 1B tokens
        )
    );

    await jetton.send(
        provider.sender(),
        { value: toNano("0.25") },
        { $$type: "Deploy", queryId: 0n }
    );

    console.log("Jetton Master deployed at:", jetton.address);

    // Mint initial supply
    await jetton.send(
        provider.sender(),
        { value: toNano("0.5") },
        {
            $$type: "JettonMint",
            receiver: owner,
            amount: toNano("1000000000"),
        }
    );

    console.log("Initial supply minted!");
}
```

### Deploy Commands
```bash
# Build contracts
npx blueprint build

# Deploy to testnet
npx blueprint run --testnet

# Deploy to mainnet
npx blueprint run --mainnet

# Run tests
npx blueprint test
```

---

## Test Template

```typescript
import { Blockchain, SandboxContract, TreasuryContract } from "@ton/sandbox";
import { toNano, Address } from "@ton/core";
import { MyJetton } from "../wrappers/MyJetton";
import "@ton/test-utils";

describe("MyJetton", () => {
    let blockchain: Blockchain;
    let deployer: SandboxContract<TreasuryContract>;
    let jetton: SandboxContract<MyJetton>;

    beforeEach(async () => {
        blockchain = await Blockchain.create();
        deployer = await blockchain.treasury("deployer");

        const content = beginCell()
            .storeUint(0x01, 8)
            .storeStringTail("https://example.com/metadata.json")
            .endCell();

        jetton = blockchain.openContract(
            await MyJetton.fromInit(
                deployer.address,
                content,
                toNano("1000000000")
            )
        );

        await jetton.send(
            deployer.getSender(),
            { value: toNano("0.25") },
            { $$type: "Deploy", queryId: 0n }
        );
    });

    it("should deploy correctly", async () => {
        // Blockchain is created and contract deployed in beforeEach
    });

    it("should mint tokens", async () => {
        const result = await jetton.send(
            deployer.getSender(),
            { value: toNano("0.5") },
            {
                $$type: "JettonMint",
                receiver: deployer.address,
                amount: toNano("1000"),
            }
        );
        expect(result.transactions).toHaveTransaction({
            from: deployer.address,
            to: jetton.address,
            success: true,
        });
    });

    it("should prevent non-owner from minting", async () => {
        const attacker = await blockchain.treasury("attacker");
        const result = await jetton.send(
            attacker.getSender(),
            { value: toNano("0.5") },
            {
                $$type: "JettonMint",
                receiver: attacker.address,
                amount: toNano("1000"),
            }
        );
        expect(result.transactions).toHaveTransaction({
            from: attacker.address,
            to: jetton.address,
            success: false,
        });
    });
});
```

---

## Cost Estimates

| Operation | ~Cost (TON) | ~Cost (USD at $5/TON) |
|-----------|-------------|----------------------|
| Jetton Master deploy | 0.1-0.3 | ~$0.50-1.50 |
| Jetton Wallet creation | 0.05 | ~$0.25 |
| NFT Collection deploy | 0.1-0.3 | ~$0.50-1.50 |
| NFT Item mint | 0.05-0.1 | ~$0.25-0.50 |
| Transfer (Jetton) | 0.05 | ~$0.25 |

TON is very cheap for deployment compared to Ethereum mainnet.

## Testnet Faucet
- **TON Testnet**: https://t.me/testgiver_ton_bot (Telegram bot)

## Explorers
- **Mainnet**: https://tonviewer.com or https://tonscan.org
- **Testnet**: https://testnet.tonviewer.com