# Ethereum / EVM Token Development Reference

## Toolchain Setup

### Hardhat (Recommended for beginners)
```bash
mkdir my-token && cd my-token
npm init -y
npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox
npx hardhat init  # Choose "Create a JavaScript project"
npm install @openzeppelin/contracts
```

### Foundry (Recommended for production)
```bash
curl -L https://foundry.paradigm.xyz | bash
foundryup
forge init my-token
forge install OpenZeppelin/openzeppelin-contracts
```

## Project Structures

### Hardhat project:
```
my-token/
├── contracts/
│   └── MyToken.sol
├── scripts/
│   └── deploy.js
├── test/
│   └── MyToken.test.js
├── hardhat.config.js
├── .env.example
├── .gitignore
└── package.json
```

### Foundry project:
```
my-token/
├── src/
│   └── MyToken.sol
├── script/
│   └── Deploy.s.sol
├── test/
│   └── MyToken.t.sol
├── foundry.toml
├── .env.example
└── .gitignore
```

---

## ERC-20 Templates

### Minimal ERC-20 (Fixed Supply)
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract MyToken is ERC20 {
    constructor(
        string memory name,
        string memory symbol,
        uint256 totalSupply
    ) ERC20(name, symbol) {
        _mint(msg.sender, totalSupply * 10 ** decimals());
    }
}
```

### ERC-20 with Mint, Burn, Pause, Access Control
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract MyToken is ERC20, ERC20Burnable, ERC20Pausable, Ownable {
    uint256 public maxSupply;

    constructor(
        string memory name,
        string memory symbol,
        uint256 initialSupply,
        uint256 _maxSupply
    ) ERC20(name, symbol) Ownable(msg.sender) {
        require(_maxSupply >= initialSupply, "Max supply must be >= initial supply");
        maxSupply = _maxSupply * 10 ** decimals();
        _mint(msg.sender, initialSupply * 10 ** decimals());
    }

    function mint(address to, uint256 amount) public onlyOwner {
        require(totalSupply() + amount <= maxSupply, "Exceeds max supply");
        _mint(to, amount);
    }

    function pause() public onlyOwner { _pause(); }
    function unpause() public onlyOwner { _unpause(); }

    function _update(address from, address to, uint256 value)
        internal override(ERC20, ERC20Pausable)
    {
        super._update(from, to, value);
    }
}
```

### ERC-20 with Tax/Fee Mechanism (Meme Coin Pattern)
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract TaxToken is ERC20, Ownable {
    uint256 public buyTaxBps;   // basis points (100 = 1%)
    uint256 public sellTaxBps;
    address public taxWallet;
    mapping(address => bool) public isExcludedFromTax;
    mapping(address => bool) public isAMM; // DEX pair addresses

    uint256 public constant MAX_TAX = 1000; // 10% max

    constructor(
        string memory name,
        string memory symbol,
        uint256 totalSupply,
        uint256 _buyTaxBps,
        uint256 _sellTaxBps,
        address _taxWallet
    ) ERC20(name, symbol) Ownable(msg.sender) {
        require(_buyTaxBps <= MAX_TAX && _sellTaxBps <= MAX_TAX, "Tax too high");
        buyTaxBps = _buyTaxBps;
        sellTaxBps = _sellTaxBps;
        taxWallet = _taxWallet;
        isExcludedFromTax[msg.sender] = true;
        isExcludedFromTax[_taxWallet] = true;
        _mint(msg.sender, totalSupply * 10 ** decimals());
    }

    function setAMM(address pair, bool value) external onlyOwner {
        isAMM[pair] = value;
    }

    function setTax(uint256 _buyTaxBps, uint256 _sellTaxBps) external onlyOwner {
        require(_buyTaxBps <= MAX_TAX && _sellTaxBps <= MAX_TAX, "Tax too high");
        buyTaxBps = _buyTaxBps;
        sellTaxBps = _sellTaxBps;
    }

    function _update(address from, address to, uint256 amount) internal override {
        if (isExcludedFromTax[from] || isExcludedFromTax[to]) {
            super._update(from, to, amount);
            return;
        }

        uint256 taxAmount = 0;
        if (isAMM[from]) { // Buy
            taxAmount = (amount * buyTaxBps) / 10000;
        } else if (isAMM[to]) { // Sell
            taxAmount = (amount * sellTaxBps) / 10000;
        }

        if (taxAmount > 0) {
            super._update(from, taxWallet, taxAmount);
        }
        super._update(from, to, amount - taxAmount);
    }
}
```

### ERC-20 Governance Token (with Votes + Timelock)
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract GovernanceToken is ERC20, ERC20Permit, ERC20Votes, Ownable {
    constructor(
        string memory name,
        string memory symbol,
        uint256 totalSupply
    ) ERC20(name, symbol) ERC20Permit(name) Ownable(msg.sender) {
        _mint(msg.sender, totalSupply * 10 ** decimals());
    }

    function _update(address from, address to, uint256 value)
        internal override(ERC20, ERC20Votes)
    {
        super._update(from, to, value);
    }

    function nonces(address owner)
        public view override(ERC20Permit, Nonces)
        returns (uint256)
    {
        return super.nonces(owner);
    }
}
```

---

## ERC-721 NFT Templates

### Basic NFT Collection
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract MyNFT is ERC721, ERC721URIStorage, Ownable {
    uint256 private _nextTokenId;
    uint256 public maxSupply;
    uint256 public mintPrice;
    bool public mintActive;

    constructor(
        string memory name,
        string memory symbol,
        uint256 _maxSupply,
        uint256 _mintPrice
    ) ERC721(name, symbol) Ownable(msg.sender) {
        maxSupply = _maxSupply;
        mintPrice = _mintPrice;
    }

    function toggleMint() external onlyOwner {
        mintActive = !mintActive;
    }

    function mint(string memory uri) public payable {
        require(mintActive, "Minting not active");
        require(msg.value >= mintPrice, "Insufficient payment");
        require(_nextTokenId < maxSupply, "Max supply reached");

        uint256 tokenId = _nextTokenId++;
        _safeMint(msg.sender, tokenId);
        _setTokenURI(tokenId, uri);
    }

    function withdraw() external onlyOwner {
        (bool success, ) = owner().call{value: address(this).balance}("");
        require(success, "Withdrawal failed");
    }

    // Required overrides
    function tokenURI(uint256 tokenId)
        public view override(ERC721, ERC721URIStorage)
        returns (string memory)
    {
        return super.tokenURI(tokenId);
    }

    function supportsInterface(bytes4 interfaceId)
        public view override(ERC721, ERC721URIStorage)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}
```

### ERC-721 with Royalties (ERC-2981)
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Royalty.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract RoyaltyNFT is ERC721, ERC721Royalty, Ownable {
    uint256 private _nextTokenId;
    string private _baseTokenURI;

    constructor(
        string memory name,
        string memory symbol,
        string memory baseURI,
        address royaltyReceiver,
        uint96 royaltyBps  // e.g., 500 = 5%
    ) ERC721(name, symbol) Ownable(msg.sender) {
        _baseTokenURI = baseURI;
        _setDefaultRoyalty(royaltyReceiver, royaltyBps);
    }

    function mint(address to) external onlyOwner {
        _safeMint(to, _nextTokenId++);
    }

    function _baseURI() internal view override returns (string memory) {
        return _baseTokenURI;
    }

    function supportsInterface(bytes4 interfaceId)
        public view override(ERC721, ERC721Royalty)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}
```

---

## ERC-1155 Semi-Fungible Template

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract GameItems is ERC1155, Ownable {
    mapping(uint256 => uint256) public maxSupply;
    mapping(uint256 => uint256) public totalMinted;
    mapping(uint256 => uint256) public mintPrice;

    constructor(string memory uri) ERC1155(uri) Ownable(msg.sender) {}

    function createItem(uint256 id, uint256 _maxSupply, uint256 _mintPrice)
        external onlyOwner
    {
        maxSupply[id] = _maxSupply;
        mintPrice[id] = _mintPrice;
    }

    function mint(uint256 id, uint256 amount) external payable {
        require(maxSupply[id] > 0, "Item does not exist");
        require(totalMinted[id] + amount <= maxSupply[id], "Exceeds max supply");
        require(msg.value >= mintPrice[id] * amount, "Insufficient payment");

        totalMinted[id] += amount;
        _mint(msg.sender, id, amount, "");
    }

    function withdraw() external onlyOwner {
        (bool success, ) = owner().call{value: address(this).balance}("");
        require(success, "Withdrawal failed");
    }
}
```

---

## DeFi Primitives

### Staking Contract
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Staking is ReentrancyGuard, Ownable {
    using SafeERC20 for IERC20;

    IERC20 public stakingToken;
    IERC20 public rewardToken;

    uint256 public rewardRate;        // tokens per second
    uint256 public lastUpdateTime;
    uint256 public rewardPerTokenStored;
    uint256 public totalStaked;

    mapping(address => uint256) public stakedBalance;
    mapping(address => uint256) public userRewardPerTokenPaid;
    mapping(address => uint256) public rewards;

    constructor(address _stakingToken, address _rewardToken, uint256 _rewardRate)
        Ownable(msg.sender)
    {
        stakingToken = IERC20(_stakingToken);
        rewardToken = IERC20(_rewardToken);
        rewardRate = _rewardRate;
    }

    modifier updateReward(address account) {
        rewardPerTokenStored = rewardPerToken();
        lastUpdateTime = block.timestamp;
        if (account != address(0)) {
            rewards[account] = earned(account);
            userRewardPerTokenPaid[account] = rewardPerTokenStored;
        }
        _;
    }

    function rewardPerToken() public view returns (uint256) {
        if (totalStaked == 0) return rewardPerTokenStored;
        return rewardPerTokenStored +
            ((block.timestamp - lastUpdateTime) * rewardRate * 1e18) / totalStaked;
    }

    function earned(address account) public view returns (uint256) {
        return (stakedBalance[account] *
            (rewardPerToken() - userRewardPerTokenPaid[account])) / 1e18
            + rewards[account];
    }

    function stake(uint256 amount) external nonReentrant updateReward(msg.sender) {
        require(amount > 0, "Cannot stake 0");
        totalStaked += amount;
        stakedBalance[msg.sender] += amount;
        stakingToken.safeTransferFrom(msg.sender, address(this), amount);
    }

    function withdraw(uint256 amount) external nonReentrant updateReward(msg.sender) {
        require(amount > 0, "Cannot withdraw 0");
        require(stakedBalance[msg.sender] >= amount, "Insufficient balance");
        totalStaked -= amount;
        stakedBalance[msg.sender] -= amount;
        stakingToken.safeTransfer(msg.sender, amount);
    }

    function claimRewards() external nonReentrant updateReward(msg.sender) {
        uint256 reward = rewards[msg.sender];
        if (reward > 0) {
            rewards[msg.sender] = 0;
            rewardToken.safeTransfer(msg.sender, reward);
        }
    }
}
```

### Vesting Contract
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

contract TokenVesting is ReentrancyGuard {
    using SafeERC20 for IERC20;

    struct VestingSchedule {
        uint256 totalAmount;
        uint256 released;
        uint256 startTime;
        uint256 cliffDuration;
        uint256 vestingDuration;
        bool revocable;
        bool revoked;
    }

    IERC20 public token;
    address public owner;
    mapping(address => VestingSchedule) public schedules;

    constructor(address _token) {
        token = IERC20(_token);
        owner = msg.sender;
    }

    function createSchedule(
        address beneficiary,
        uint256 totalAmount,
        uint256 startTime,
        uint256 cliffDuration,
        uint256 vestingDuration,
        bool revocable
    ) external {
        require(msg.sender == owner, "Not owner");
        require(schedules[beneficiary].totalAmount == 0, "Schedule exists");

        schedules[beneficiary] = VestingSchedule({
            totalAmount: totalAmount,
            released: 0,
            startTime: startTime,
            cliffDuration: cliffDuration,
            vestingDuration: vestingDuration,
            revocable: revocable,
            revoked: false
        });

        token.safeTransferFrom(msg.sender, address(this), totalAmount);
    }

    function release() external nonReentrant {
        VestingSchedule storage schedule = schedules[msg.sender];
        require(schedule.totalAmount > 0, "No schedule");
        require(!schedule.revoked, "Revoked");

        uint256 releasable = _vestedAmount(schedule) - schedule.released;
        require(releasable > 0, "Nothing to release");

        schedule.released += releasable;
        token.safeTransfer(msg.sender, releasable);
    }

    function _vestedAmount(VestingSchedule memory schedule)
        internal view returns (uint256)
    {
        if (block.timestamp < schedule.startTime + schedule.cliffDuration) {
            return 0;
        }
        if (block.timestamp >= schedule.startTime + schedule.vestingDuration) {
            return schedule.totalAmount;
        }
        return (schedule.totalAmount *
            (block.timestamp - schedule.startTime)) / schedule.vestingDuration;
    }
}
```

---

## Deployment Scripts

### Hardhat Deploy Script
```javascript
const { ethers } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with:", deployer.address);
  console.log("Balance:", ethers.formatEther(await ethers.provider.getBalance(deployer.address)));

  const Token = await ethers.getContractFactory("MyToken");
  const token = await Token.deploy(
    "My Token",           // name
    "MTK",                // symbol
    1000000               // total supply (before decimals)
  );

  await token.waitForDeployment();
  const address = await token.getAddress();
  console.log("Token deployed to:", address);

  // Wait for block confirmations then verify
  console.log("Waiting for confirmations...");
  await token.deploymentTransaction().wait(5);

  console.log("Verifying on Etherscan...");
  await hre.run("verify:verify", {
    address: address,
    constructorArguments: ["My Token", "MTK", 1000000],
  });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

### Hardhat Config
```javascript
require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

module.exports = {
  solidity: {
    version: "0.8.20",
    settings: { optimizer: { enabled: true, runs: 200 } }
  },
  networks: {
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL || "https://rpc.sepolia.org",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
    },
    mainnet: {
      url: process.env.MAINNET_RPC_URL || "",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
    },
    polygon: {
      url: process.env.POLYGON_RPC_URL || "https://polygon-rpc.com",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
    },
    bsc: {
      url: process.env.BSC_RPC_URL || "https://bsc-dataseed.binance.org",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
    },
    base: {
      url: process.env.BASE_RPC_URL || "https://mainnet.base.org",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
    },
    arbitrum: {
      url: process.env.ARBITRUM_RPC_URL || "https://arb1.arbitrum.io/rpc",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
    },
  },
  etherscan: {
    apiKey: process.env.ETHERSCAN_API_KEY || "",
  },
};
```

### .env.example
```
PRIVATE_KEY=your_private_key_here_NEVER_commit_this
SEPOLIA_RPC_URL=https://rpc.sepolia.org
MAINNET_RPC_URL=
ETHERSCAN_API_KEY=
```

---

## Test Template (Hardhat)

```javascript
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("MyToken", function () {
  let token, owner, addr1, addr2;

  beforeEach(async function () {
    [owner, addr1, addr2] = await ethers.getSigners();
    const Token = await ethers.getContractFactory("MyToken");
    token = await Token.deploy("My Token", "MTK", 1000000);
    await token.waitForDeployment();
  });

  describe("Deployment", function () {
    it("Should set the right name and symbol", async function () {
      expect(await token.name()).to.equal("My Token");
      expect(await token.symbol()).to.equal("MTK");
    });

    it("Should mint total supply to deployer", async function () {
      const total = await token.totalSupply();
      expect(await token.balanceOf(owner.address)).to.equal(total);
    });
  });

  describe("Transfers", function () {
    it("Should transfer tokens between accounts", async function () {
      await token.transfer(addr1.address, 1000n * 10n ** 18n);
      expect(await token.balanceOf(addr1.address)).to.equal(1000n * 10n ** 18n);
    });

    it("Should fail if sender doesn't have enough tokens", async function () {
      await expect(
        token.connect(addr1).transfer(owner.address, 1)
      ).to.be.reverted;
    });
  });
});
```

---

## Gas Cost Estimates (2025-2026 averages)

| Operation | Estimated Gas | ~Cost at 20 gwei |
|-----------|---------------|-------------------|
| ERC-20 deploy (simple) | ~800K | ~$4-8 |
| ERC-20 deploy (full features) | ~1.5M | ~$8-15 |
| ERC-721 deploy | ~2M | ~$10-20 |
| ERC-1155 deploy | ~1.8M | ~$9-18 |
| Staking contract deploy | ~1.2M | ~$6-12 |

Note: L2s (Base, Arbitrum, Optimism) are 10-50x cheaper. BSC and Polygon are also significantly cheaper.

## Testnet Faucets

- **Sepolia ETH**: https://sepoliafaucet.com or https://www.alchemy.com/faucets/ethereum-sepolia
- **Mumbai MATIC**: https://faucet.polygon.technology
- **BSC Testnet**: https://testnet.bnbchain.org/faucet-smart
- **Base Sepolia**: https://www.coinbase.com/faucets/base-ethereum-goerli-faucet
