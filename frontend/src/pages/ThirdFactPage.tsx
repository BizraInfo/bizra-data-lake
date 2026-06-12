import { type ReactNode, useEffect, useState } from 'react';
import '../styles/third-fact.css';

type EvidenceItem = {
  value: string;
  label: string;
  href?: string;
  source: string;
  secondary?: boolean;
};

type Pillar = {
  number: string;
  name: string;
  arabic: string;
  description: string;
  full?: boolean;
};

type LadderStep = {
  key: string;
  description: string;
  active?: boolean;
  future?: boolean;
};

type SectionNavItem = {
  id: string;
  label: string;
};

const NAV_ITEMS: SectionNavItem[] = [
  { id: 'third-fact-hero', label: 'The Seed' },
  { id: 'third-fact-declaration', label: 'Opening Declaration' },
  { id: 'third-fact-question', label: 'The Question Nobody Is Asking' },
  { id: 'third-fact-core', label: 'The Third Fact' },
  { id: 'third-fact-architecture', label: 'What BIZRA Actually Is' },
  { id: 'third-fact-pillars', label: 'The Seven Brand Pillars' },
  { id: 'third-fact-urp', label: 'The Universal Resource Pool' },
  { id: 'third-fact-proof', label: 'Proof, Not Assumption' },
  { id: 'third-fact-economy', label: 'The Economy of Verified Impact' },
  { id: 'third-fact-mission', label: 'The Human Mission Is the Center' },
  { id: 'third-fact-node0', label: 'Node0' },
  { id: 'third-fact-verification', label: 'What We Are Asking' },
];

const DEBT_EVIDENCE: EvidenceItem[] = [
  {
    value: '$102T',
    label: 'Global public debt reached in 2024',
    href: 'https://unctad.org/news/global-public-debt-hits-record-102-trillion-2024-striking-developing-countries-hardest',
    source: 'UNCTAD · UN Trade & Development',
  },
  {
    value: '3.4B',
    label: 'People in countries paying more on debt than health or education',
    href: 'https://unctad.org/news/global-public-debt-hits-record-102-trillion-2024-striking-developing-countries-hardest',
    source: 'UNCTAD · UN Trade & Development',
  },
  {
    value: '$318T',
    label: 'Total global debt · ~328% of GDP',
    href: 'https://www.aa.com.tr/en/economy/global-debt-climbs-to-record-318t-in-2024/3493336',
    source: 'IIF Global Debt Monitor · via AA',
    secondary: true,
  },
];

const AI_EVIDENCE: EvidenceItem[] = [
  {
    value: '945 TWh',
    label: 'Projected data-centre electricity consumption by 2030',
    href: 'https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai',
    source: 'IEA · Energy and AI',
  },
  {
    value: '$400B+',
    label: 'CapEx by 5 large tech firms in 2025, rising further in 2026',
    href: 'https://www.iea.org/news/data-centre-electricity-use-surged-in-2025-even-with-tightening-bottlenecks-driving-a-scramble-for-solutions',
    source: 'IEA · Data Centre Report',
  },
  {
    value: 'Few',
    label: 'Entities own the infrastructure all of humanity depends on',
    source: 'BIZRA · Pattern Recognition',
  },
];

const ARCH_ROWS = [
  ['PAT', '7 Personal Agentic Team agents — serve the human sovereign directly, privately'],
  ['SAT', '5 System Agentic Team agents — serve the larger ecosystem, bound by constitution'],
  ['DEMA', 'The visible bridge — trusted companion interface between heart, mind, and action'],
  ['FATE', 'Constitutional boundary gate — no action crosses without consent and proof'],
  ['URP', 'Universal Resource Pool seed — the shared substrate of the ecosystem'],
  ['RECEIPTS', 'Tamper-evident, hash-chained, replayable records of verified action'],
  ['POI', 'Proof of Impact ledger — verified contribution scores that govern reward eligibility'],
] as const;

const PILLARS: Pillar[] = [
  {
    number: 'PILLAR 01',
    name: 'Meaning',
    arabic: 'المعنى',
    description: 'Words are not disposable. A claim must carry evidence. A promise must carry consequence.',
  },
  {
    number: 'PILLAR 02',
    name: 'Humility',
    arabic: 'الخشوع الحقيقي',
    description: 'Knowledge should increase humility, not arrogance. BIZRA speaks with confidence only where it has proof.',
  },
  {
    number: 'PILLAR 03',
    name: 'Proof',
    arabic: 'البرهان',
    description: 'The system moves toward receipts, traceability, replayability, and verified action.',
  },
  {
    number: 'PILLAR 04',
    name: 'Ihsan',
    arabic: 'الإحسان',
    description: 'Excellence with conscience. Precision without cruelty. Power without corruption.',
  },
  {
    number: 'PILLAR 05',
    name: 'Sovereignty',
    arabic: 'السيادة',
    description: 'The human is not raw material for a platform. The user owns their node, data, keys, mission, and path.',
  },
  {
    number: 'PILLAR 06',
    name: 'Growth',
    arabic: 'النماء',
    description: 'A seed does not dominate. It grows, adapts to soil, seeks light, and gives fruit.',
  },
  {
    number: 'PILLAR 07',
    name: 'Mercy and Peace',
    arabic: 'الرحمة والسلام',
    description: 'BIZRA calls people back to dignity, solidarity, kindness, and support. The forest does not grow by destroying its own trees.',
    full: true,
  },
];

const URP_LADDER: LadderStep[] = [
  {
    key: 'URP_LOCAL_ACTIVE · Current Stage',
    description: 'Node0 alone. The seed is planted. The pattern is complete. The receipts begin.',
    active: true,
  },
  {
    key: 'PRIVATE_PILOT_URP',
    description: 'Node0 and Node1 federate only after local readiness is proven. Trust is established between two proven nodes.',
  },
  {
    key: 'PILOT_SHARED_URP',
    description: 'Three to five trusted nodes. Shared receipts cross-validate. Direction only — not yet proven.',
    future: true,
  },
  {
    key: 'UNIVERSAL_NETWORK_URP',
    description: 'Only after legal, technical, security, and social validation. Direction only — not yet proven.',
    future: true,
  },
];

const NODE_LADDER: LadderStep[] = [
  {
    key: 'SEED · Genesis Activation · Current',
    description: 'Identity creation. Constitution acceptance. Receipt chain initialization. Node0 alone, complete.',
    active: true,
  },
  {
    key: 'SPROUT · System Initialization',
    description: 'PAT/SAT initialization. Knowledge ingestion. First validations. Receipts accumulate.',
  },
  {
    key: 'TREE · Federation · Direction',
    description: 'Federation handshake and verified sharing between proven nodes. The forest begins.',
    future: true,
  },
  {
    key: 'FOREST · Propagation · Direction',
    description: 'New node seeding, knowledge propagation, and Proof-of-Impact attestations. Direction only.',
    future: true,
  },
];

const REFERENCES = [
  {
    label: 'UNCTAD — Global public debt hit a record $102 trillion in 2024',
    href: 'https://unctad.org/news/global-public-debt-hits-record-102-trillion-2024-striking-developing-countries-hardest',
  },
  {
    label: 'IEA — Energy demand from AI: Energy and AI Analysis',
    href: 'https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai',
  },
  {
    label: 'IEA — Data centre electricity use surged in 2025',
    href: 'https://www.iea.org/news/data-centre-electricity-use-surged-in-2025-even-with-tightening-bottlenecks-driving-a-scramble-for-solutions',
  },
  {
    label: 'Anadolu Ajansi — Global debt climbs to record $318T in 2024',
    href: 'https://www.aa.com.tr/en/economy/global-debt-climbs-to-record-318t-in-2024/3493336',
  },
  {
    label: 'OECD — Global Debt Report 2025',
    href: 'https://www.oecd.org/en/publications/global-debt-report-2025_8ee42b13-en.html',
  },
];

function ensureMeta(attribute: 'name' | 'property', key: string, content: string) {
  let element = document.querySelector<HTMLMetaElement>(`meta[${attribute}="${key}"]`);
  if (!element) {
    element = document.createElement('meta');
    element.setAttribute(attribute, key);
    document.head.appendChild(element);
  }
  element.setAttribute('content', content);
}

function ensureCanonical(href: string) {
  let element = document.querySelector<HTMLLinkElement>('link[rel="canonical"]');
  if (!element) {
    element = document.createElement('link');
    element.setAttribute('rel', 'canonical');
    document.head.appendChild(element);
  }
  element.setAttribute('href', href);
}

function useThirdFactHead() {
  useEffect(() => {
    const previousTitle = document.title;
    document.title = 'BIZRA — The Third Fact: Humanity Is the Infrastructure';
    ensureMeta('name', 'description', 'BIZRA is a seed architecture for sovereign, distributed, constitutional intelligence. The Third Fact: humanity is not the fuel — humanity is the infrastructure.');
    ensureMeta('property', 'og:type', 'article');
    ensureMeta('property', 'og:title', 'BIZRA — The Third Fact: Humanity Is the Infrastructure');
    ensureMeta('property', 'og:description', 'A manifesto for human sovereignty in the age of artificial intelligence, debt, and concentrated power.');
    ensureMeta('property', 'og:url', `${window.location.origin}/third-fact`);
    ensureMeta('name', 'twitter:card', 'summary_large_image');
    ensureMeta('name', 'twitter:title', 'BIZRA — The Third Fact: Humanity Is the Infrastructure');
    ensureMeta('name', 'twitter:description', 'A manifesto for human sovereignty in the age of artificial intelligence, debt, and concentrated power.');
    ensureCanonical(`${window.location.origin}/third-fact`);

    return () => {
      document.title = previousTitle;
    };
  }, []);
}

function useReadingProgress() {
  const [progress, setProgress] = useState(0);
  const [activeSection, setActiveSection] = useState(NAV_ITEMS[0].id);

  useEffect(() => {
    const onScroll = () => {
      const documentElement = document.documentElement;
      const total = Math.max(documentElement.scrollHeight - documentElement.clientHeight, 1);
      const nextProgress = Math.min(100, Math.max(0, Math.round((documentElement.scrollTop / total) * 100)));
      setProgress(nextProgress);

      const probe = window.scrollY + window.innerHeight * 0.4;
      for (let index = NAV_ITEMS.length - 1; index >= 0; index -= 1) {
        const section = document.getElementById(NAV_ITEMS[index].id);
        if (section && probe >= section.offsetTop) {
          setActiveSection(NAV_ITEMS[index].id);
          break;
        }
      }
    };

    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return { activeSection, progress };
}

function EvidenceBlock({ title, items }: { title: string; items: EvidenceItem[] }) {
  return (
    <>
      <div className="tf-receipt-label">{title}</div>
      <div className="tf-evidence-block" role="region" aria-label={title}>
        {items.map(item => (
          <div className="tf-evidence-cell" key={`${item.value}-${item.source}`}>
            <span className="tf-evidence-number">{item.value}</span>
            <div className="tf-evidence-label">
              {item.label}
              {item.secondary && <span className="tf-secondary-flag">secondary</span>}
            </div>
            {item.href ? (
              <a className="tf-evidence-cite" href={item.href} target="_blank" rel="noopener noreferrer">
                {item.source} ↗
              </a>
            ) : (
              <span className="tf-evidence-cite tf-static-cite">{item.source}</span>
            )}
          </div>
        ))}
      </div>
    </>
  );
}

function SectionHeader({ number, title }: { number: string; title: ReactNode }) {
  return (
    <>
      <div className="tf-section-header" aria-hidden="true">
        <span className="tf-section-number">{number}</span>
        <div className="tf-section-rule" />
      </div>
      <h2 className="tf-section-title">{title}</h2>
    </>
  );
}

function Verse({ lines }: { lines: ReactNode[] }) {
  return (
    <div className="tf-verse">
      {lines.map((line, index) => (
        <span className="tf-verse-line" key={index}>{line}</span>
      ))}
    </div>
  );
}

function PullQuote({ children, arabic }: { children: ReactNode; arabic: string }) {
  return (
    <blockquote className="tf-pull-quote">
      <p className="tf-pull-quote-text">{children}</p>
      <p className="tf-pull-quote-arabic" lang="ar" dir="rtl">{arabic}</p>
    </blockquote>
  );
}

function ArchitectureRows({ rows }: { rows: readonly (readonly [string, string])[] }) {
  return (
    <div className="tf-arch-block">
      <div className="tf-arch-title">Node Architecture · Every Human Node Contains</div>
      {rows.map(([key, value]) => (
        <div className="tf-arch-row" key={key}>
          <div className="tf-arch-key">{key}</div>
          <div className="tf-arch-value">{value}</div>
        </div>
      ))}
    </div>
  );
}

function Ladder({ steps }: { steps: LadderStep[] }) {
  return (
    <div className="tf-ladder" role="list">
      {steps.map(step => (
        <div className={`tf-ladder-step${step.active ? ' tf-active' : ''}${step.future ? ' tf-future' : ''}`} role="listitem" key={step.key}>
          <div className="tf-ladder-icon" aria-hidden="true"><div className="tf-ladder-dot" /></div>
          <div>
            <div className="tf-ladder-key">{step.key}</div>
            <div className="tf-ladder-desc">{step.description}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function ThirdFactPage() {
  useThirdFactHead();
  const { activeSection, progress } = useReadingProgress();

  return (
    <article className="third-fact-page">
      <a href="#third-fact-main" className="tf-skip-link">Skip to main content</a>
      <div
        className="tf-progress-bar"
        role="progressbar"
        aria-label="Reading progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progress}
        style={{ width: `${progress}%` }}
      />
      <nav className="tf-side-nav" aria-label="Section navigation">
        {NAV_ITEMS.map(item => (
          <button
            className={`tf-nav-dot${activeSection === item.id ? ' tf-active' : ''}`}
            key={item.id}
            type="button"
            aria-label={`Jump to: ${item.label}`}
            title={item.label}
            onClick={() => document.getElementById(item.id)?.scrollIntoView({ behavior: 'smooth' })}
          />
        ))}
      </nav>

      <header className="tf-hero" id="third-fact-hero">
        <svg className="tf-seed-canvas" viewBox="-60 -60 120 120" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
          <circle className="tf-outer-pulse" cx="0" cy="0" r="54" />
          <circle cx="0" cy="0" r="18" />
          <circle cx="18" cy="0" r="18" />
          <circle cx="9" cy="15.6" r="18" />
          <circle cx="-9" cy="15.6" r="18" />
          <circle cx="-18" cy="0" r="18" />
          <circle cx="-9" cy="-15.6" r="18" />
          <circle cx="9" cy="-15.6" r="18" />
          <circle className="tf-nuqta" cx="0" cy="0" r="1.2" />
        </svg>

        <div className="tf-hero-content">
          <p className="tf-hero-overline">Node0 · Third Fact Public Document v0.1 · Claim Discipline Active</p>
          <div className="tf-hero-arabic" lang="ar" dir="rtl">البذرة</div>
          <h1 className="tf-hero-name">BIZRA<span>The Seed of Sovereign Intelligence</span></h1>
          <p className="tf-hero-tagline">A Manifesto for Human Sovereignty in the Age of Artificial Intelligence, Debt, and Concentrated Power</p>
          <div className="tf-hero-divider" aria-hidden="true" />
          <p className="tf-hero-subtitle">Before the code. Before the architecture. Before the model. There was a word. A belief. A seed. This is what it grew into — and why it matters to every human on Earth.</p>
          <p className="tf-hero-byline">By the First Architect of BIZRA — Node0 · Ramadan 2023 → 2026</p>
        </div>
      </header>

      <main className="tf-paper" id="third-fact-main">
        <section className="tf-declaration" id="third-fact-declaration" aria-labelledby="third-fact-declaration-title">
          <div className="tf-claim-badge"><span>◆</span> Claim Discipline Active | <span>◆</span> No Unverified Technical Claims</div>
          <h2 className="tf-sr-only" id="third-fact-declaration-title">Opening Declaration</h2>
          <p className="tf-declaration-text">I did not come from an institution.</p>
          <p className="tf-declaration-text">I did not come from a research lab with billions behind me. I did not come from a fund, a government, or a global technology empire.</p>
          <Verse lines={[
            <>I came with something <strong>older than capital</strong> and stronger than permission.</>,
            'A belief.',
            'A word.',
            <strong>A seed.</strong>,
          ]} />
          <p className="tf-declaration-text">For three years, I worked alone on BIZRA every single day — not because it was easy, but because I saw a pattern I could not <em>unsee.</em></p>
          <p className="tf-lead">The world is building the next layer of intelligence on top of the same architecture that broke the last layer of trust: <strong>concentration, extraction, debt, opacity, and control.</strong></p>
        </section>

        <section className="tf-section" id="third-fact-question" aria-labelledby="third-fact-question-title">
          <SectionHeader number="§ I" title={<span id="third-fact-question-title">The Question<br />Nobody Is Asking</span>} />
          <p>Ask the market what is needed to build better AI, and the answer is almost always the same: <strong>data and compute.</strong> Then ask the deeper question:</p>
          <Verse lines={['Who owns the data?', 'Who owns the compute?', 'Who owns the model?', 'Who owns the infrastructure?', 'Who receives the value?']} />
          <p>In the current path, the answer is not humanity. It is a small number of centralized powers. The danger is not artificial intelligence itself — the danger is building a future where billions of humans become raw material for intelligence they helped create but do not own.</p>
          <p>This is not only an AI problem. It is the same pattern that appears in the global financial system — a pattern that has already proven catastrophic at scale.</p>
          <EvidenceBlock title="Evidence Receipt · Global Debt · Verified Citations" items={DEBT_EVIDENCE} />
          <p>The OECD warns that already-high debt must shift from merely supporting recovery toward financing real investment — under conditions of slowing growth, rising geopolitical risk, and higher long-term rates. <a className="tf-inline-link" href="https://www.oecd.org/en/publications/global-debt-report-2025_8ee42b13-en.html" target="_blank" rel="noopener noreferrer">OECD Global Debt Report 2025 ↗</a></p>
          <PullQuote arabic="هل يجب على نظام الذكاء المستقبلي أن يكرر نفس المعمار؟">If the debt-based system already struggles under the weight of promises it cannot repay — should the intelligence system of the future repeat the same architecture?</PullQuote>
          <EvidenceBlock title="Evidence Receipt · AI Infrastructure Scale · Verified Citations" items={AI_EVIDENCE} />
        </section>

        <section className="tf-section" id="third-fact-core" aria-labelledby="third-fact-core-title">
          <SectionHeader number="§ II" title={<span id="third-fact-core-title">The <em>Third</em> Fact</span>} />
          <p>The <strong>first fact</strong> is that intelligence needs computation. Modern AI runs on chips, electricity, storage, cooling, networking, data centers, and vast capital.</p>
          <p>The <strong>second fact</strong> is that intelligence needs data — human language, behavior, creativity, memory, code, questions, pain, and dreams. The data of intelligence comes from life.</p>
          <p>But here is the fracture: <strong>the data comes from the many, while the infrastructure is owned by the few.</strong> Web2 captured community value inside platforms. Web3 often repeated another mistake: tokens before utility, speculation before service, hype before proof.</p>
          <div className="tf-law-block">
            <div className="tf-law-mono">The Third Fact</div>
            <div className="tf-law-arabic" lang="ar" dir="rtl">الإنسانية ليست وقوداً. الإنسانية هي البنية التحتية.</div>
            <p className="tf-law-english"><strong>Humanity already contains</strong> the compute, the data, the intention, the labor, the knowledge, the creativity, and the moral stake required to build another path.</p>
          </div>
          <Verse lines={['A laptop here. A phone there.', "A student's curiosity. A developer's skill.", "A mother's memory. A farmer's practical wisdom.", "A teacher's patience. A doctor's experience.", "A believer's sincerity. A builder's discipline.", <strong>A community's need.</strong>]} />
          <p>The future of ethical intelligence should be grown <strong>like a forest</strong> — not concentrated into distant industrial temples.</p>
          <PullQuote arabic="كل إنسان عقدة. وكل عقدة بذرة. وكل بذرة لها إمكانية لا نهائية.">Every human can become a node.<br />Every node can become a seed.<br />Every seed can contribute to the forest.</PullQuote>
          <p>This is not a slogan. It is the core constitutional idea of BIZRA: a node is a sovereign human plus chosen substrate, holding authority over data, agents, and consent.</p>
          <p className="tf-statement">The human is the node.<br />The machine is only the substrate.</p>
        </section>

        <section className="tf-section" id="third-fact-architecture" aria-labelledby="third-fact-architecture-title">
          <SectionHeader number="§ III" title={<span id="third-fact-architecture-title">What BIZRA<br /><em>Actually</em> Is</span>} />
          <div className="tf-section-arabic" lang="ar" dir="rtl">نظام ذكاء سيادي موزع ودستوري</div>
          <p>BIZRA is a <strong>sovereign, distributed, constitutional intelligence ecosystem.</strong> Not merely a chatbot, a token, a blockchain, an operating system, or a model. BIZRA is a seed architecture for a different relationship between humans, intelligence, resources, proof, and value.</p>
          <ArchitectureRows rows={ARCH_ROWS} />
          <p>An AI ecosystem without proof becomes <em>persuasion.</em> A token ecosystem without proof becomes <em>speculation.</em> A social ecosystem without proof becomes <em>noise.</em> A financial ecosystem without justice becomes <em>extraction.</em></p>
          <p>BIZRA is an attempt to bind intelligence back to <strong>evidence, consent, and impact.</strong></p>
        </section>

        <section className="tf-section" id="third-fact-pillars" aria-labelledby="third-fact-pillars-title">
          <SectionHeader number="§ IV" title={<span id="third-fact-pillars-title">The Seven<br />Brand Pillars</span>} />
          <p>BIZRA's character is not a marketing choice. It is system physics — the operating principles encoded into every interface, every agent response, every receipt, every claim.</p>
          <div className="tf-pillars-grid" role="list">
            {PILLARS.map(pillar => (
              <div className={`tf-pillar${pillar.full ? ' tf-pillar-full' : ''}`} role="listitem" key={pillar.number}>
                <div className="tf-pillar-number">{pillar.number}</div>
                <div className="tf-pillar-name">{pillar.name}</div>
                <div className="tf-pillar-arabic" lang="ar" dir="rtl">{pillar.arabic}</div>
                <div className="tf-pillar-desc">{pillar.description}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="tf-section" id="third-fact-urp" aria-labelledby="third-fact-urp-title">
          <SectionHeader number="§ V" title={<span id="third-fact-urp-title">The Universal<br />Resource Pool</span>} />
          <p>The Universal Resource Pool — URP — is the soil of BIZRA. Not only a compute pool. Not only storage. Not only a future network.</p>
          <p>URP is the shared substrate where resources, reusable skills, knowledge packs, receipts, proofs, SAT capabilities, contribution records, and Proof-of-Impact events become <strong>visible, verifiable, and reusable.</strong></p>
          <p>URP starts from Node0. BIZRA does not wait for a million users before it becomes alive. The first node already contains the complete seed pattern. From there, the truth ladder grows — not by claim, but by proof:</p>
          <Ladder steps={URP_LADDER} />
          <p className="tf-statement">We do not claim the forest before we prove the seed.</p>
        </section>

        <section className="tf-section" id="third-fact-proof" aria-labelledby="third-fact-proof-title">
          <SectionHeader number="§ VI" title={<span id="third-fact-proof-title">Proof,<br />Not Assumption</span>} />
          <p>A dangerous world is a world where assumption becomes the source of truth. The foundational mindset of BIZRA is drawn from a deep intellectual tradition:</p>
          <div className="tf-law-block">
            <div className="tf-law-mono">Foundational Mindset · The Law of Assumption</div>
            <div className="tf-law-arabic" lang="ar" dir="rtl">كلما ازددت علماً، ازددت يقيناً بجهلي. لا نفترض ولا نقبل الظن المجرد. وإذا كان الافتراض أمراً لا مفر منه، فإننا نفترض بإحسان.</div>
            <p className="tf-law-english">The more I learn, the more certain I become of my ignorance. <strong>We do not assume blindly. And when assumption is unavoidable, we declare the boundary between evidence and uncertainty.</strong></p>
          </div>
          <div className="tf-arch-block">
            <div className="tf-arch-title">Third Fact Protocol · Proof Chain</div>
            {[
              ['MIND', 'May propose'],
              ['MEMORY', 'May retrieve'],
              ['LOGIC', 'Must test'],
              ['CRYPTO', 'Must seal'],
              ['RECEIPTS', 'Must preserve — tamper-evident, replayable, hash-chained records'],
              ['HUMAN', 'Sovereign must consent'],
            ].map(([key, value]) => (
              <div className="tf-arch-row" key={key}><div className="tf-arch-key">{key}</div><div className="tf-arch-value">{value}</div></div>
            ))}
          </div>
          <p>This is why BIZRA does not treat intelligence as only generation. An answer is not enough. A prediction is not enough. A claim is not enough. A token is not enough. A mission is not enough.</p>
          <p className="tf-question">Where is the proof?</p>
        </section>

        <section className="tf-section" id="third-fact-economy" aria-labelledby="third-fact-economy-title">
          <SectionHeader number="§ VII" title={<span id="third-fact-economy-title">The Economy<br />of Verified Impact</span>} />
          <p>If BIZRA has an economy, it must not reward noise, speculation, empty activity, or leverage simply because leverage can buy influence.</p>
          <div className="tf-econ-chain" role="img" aria-label="Economic chain: Contribution to Verification to Receipt to Impact Score to Reward Eligibility">
            {['Contribution', 'Verification', 'Receipt', 'Impact Score', 'Reward Eligibility'].map((node, index) => (
              <span className="tf-econ-group" key={node}>
                <span className="tf-econ-node">{node}</span>
                {index < 4 && <span className="tf-econ-arrow" aria-hidden="true">→</span>}
              </span>
            ))}
          </div>
          <p>This is where BIZRA connects technology with Islamic financial principles — not as decoration, but as <strong>system physics.</strong></p>
          <Verse lines={['No riba as an extraction engine.', 'No fake value without productive grounding.', 'No hidden uncertainty without disclosure.', 'No reward detached from real benefit.', 'No economy that grows by crushing the weak.']} />
          <p>The aim is to build a micro-economy where <strong>value follows verified benefit.</strong> No reward without audit trail. The rule: <em>Verified useful impact may earn reward.</em></p>
        </section>

        <section className="tf-section" id="third-fact-mission" aria-labelledby="third-fact-mission-title">
          <SectionHeader number="§ VIII" title={<span id="third-fact-mission-title">The Human Mission<br />Is the Center</span>} />
          <p>The current AI race puts the model at the center. BIZRA does not.</p>
          <div className="tf-arch-block">
            <div className="tf-arch-title">The Shift</div>
            <div className="tf-arch-row"><div className="tf-arch-key tf-muted">FROM</div><div className="tf-arch-value tf-muted">Model-centric · centralized · extractive · assumption · debt-pressure · platform ownership</div></div>
            <div className="tf-arch-row"><div className="tf-arch-key">TO</div><div className="tf-arch-value tf-gold">Mission-centric · distributed · contributive · receipt · impact-value · human sovereignty</div></div>
          </div>
          <p>In BIZRA, the human mission is the center. The model is a tool. The agent is a servant. The node is sovereign. The receipt is witness. The constitution is boundary. The URP is commons. The reward follows impact.</p>
          <p>This is not a guaranteed AGI path. It is a disciplined attempt to explore an ethical intelligence ecosystem without repeating the mistakes of Web2 extraction, Web3 speculation, and debt-driven concentration.</p>
        </section>

        <section className="tf-section" id="third-fact-node0" aria-labelledby="third-fact-node0-title">
          <SectionHeader number="§ IX" title={<span id="third-fact-node0-title">Node0</span>} />
          <p>Every world begins somewhere. BIZRA begins with Node0. Not as mythology. Not as ego. Not as a claim of perfection.</p>
          <p className="tf-statement">Node0 is proof of origin.</p>
          <Verse lines={['One human. Two personal devices.', 'A local runtime. Seven private agents. Five system agents.', 'A local URP seed. A proof ledger.', 'A mission loop. A receipt chain.', <strong>A willingness to start before the world understands.</strong>]} />
          <Ladder steps={NODE_LADDER} />
          <p>Not a jump to a million nodes. A seed. A second node. A small pilot. A verified loop. A stronger proof. Then growth.</p>
        </section>

        <section className="tf-section" id="third-fact-verification" aria-labelledby="third-fact-verification-title">
          <SectionHeader number="§ X" title={<span id="third-fact-verification-title">What We<br />Are Asking</span>} />
          <p>BIZRA is not asking humanity to trust a founder. It is asking humanity to <em>verify a path.</em></p>
          <Verse lines={['Do not believe the words alone.', 'Test the node. Inspect the receipts.', 'Run the proof. Review the code.', 'Challenge the claims. Measure the impact.', 'Check the boundary.']} />
          <PullQuote arabic="إن أخطأنا صحّحنا. وإن ثبتنا بنينا. وإن نفعنا شاركنا. وإن أضررنا توقفنا.">If it fails — we correct it.<br />If it proves — we build on it.<br />If it helps — we share it.<br />If it harms — we stop it.</PullQuote>
          <p>This is Ihsan in engineering form. <strong>Excellence is honesty under pressure.</strong></p>
        </section>

        <section className="tf-closing" id="third-fact-closing" aria-labelledby="third-fact-closing-title">
          <div className="tf-law-mono" id="third-fact-closing-title">Closing Declaration · Node0 · BIZRA</div>
          <p className="tf-closing-line">I am not here to say I have solved the world.</p>
          <p className="tf-closing-line">I am here to say <em>I refuse</em> to accept that the future must be owned by the few.</p>
          <p className="tf-closing-line">I refuse to accept that intelligence must become another rented dependency.</p>
          <p className="tf-closing-line">I refuse to accept that technology must choose between power and mercy.</p>
          <p className="tf-closing-line tf-bold">BIZRA begins with one seed.</p>
          <p className="tf-closing-line">But one seed is not the forest. The forest comes when others plant.</p>
          <p className="tf-closing-line tf-gold">It will be because people believed another path was possible — and then proved it together.</p>
          <div className="tf-final-statement">
            <div className="tf-final-arabic" lang="ar" dir="rtl">كل إنسان عقدة.<br />وكل عقدة بذرة.<br />وكل بذرة لها إمكانية لا نهائية.<br />وكل مساهمة موثقة يمكن أن تصبح نوراً للغابة كلها.</div>
            <p className="tf-final-english">Every human is a node.<br />Every node is a seed.<br />Every seed has infinite potential.<br />And every verified contribution can become light for the whole forest.</p>
          </div>
          <PullQuote arabic="الإنسانية ليست وقوداً. الإنسانية هي البنية التحتية.">Humanity is not the fuel.<br /><strong>Humanity is the infrastructure.</strong></PullQuote>
          <p className="tf-document-end">This is the Third Fact.<br />This is BIZRA.</p>
          <div className="tf-final-arabic" lang="ar" dir="rtl">هذه هي البذرة. هذه هي بذرة.</div>
        </section>
      </main>

      <footer className="tf-footer">
        <div className="tf-footer-logo">BIZRA</div>
        <div className="tf-footer-arabic" lang="ar" dir="rtl">البذرة</div>
        <div className="tf-footer-canon">
          The Seed of Sovereign Intelligence<br />
          Brand Identity Canon v0.2 · Third Fact Public Document v0.1<br />
          Claim Discipline Active · No Unverified Technical Claims Published<br />
          Direction-only language used for future URP stages and unverified capabilities<br />
          Node0 · Ramadan 2023 → 2026 · BIZRA
        </div>
        <div className="tf-reference-wrap">
          <p className="tf-reference-title">References</p>
          <ol className="tf-reference-list">
            {REFERENCES.map(reference => (
              <li key={reference.href}><a href={reference.href} target="_blank" rel="noopener noreferrer">{reference.label} ↗</a></li>
            ))}
          </ol>
        </div>
      </footer>
    </article>
  );
}
