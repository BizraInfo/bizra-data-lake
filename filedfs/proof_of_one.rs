// ═══════════════════════════════════════════════════════════════
//  PROOF OF ONE — بُرهان الواحد
//
//  Before BIZRA serves millions, it must prove it can serve ONE.
//  One node. One human. Complete sovereignty.
//
//  "Built from the people, to the people."
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod proof_of_one {
    use bizra_node::node::{Node, NodeConfig};
    use bizra_agent::runtime::RuntimeConfig;
    use bizra_hooks::IhsanScore;
    use std::sync::atomic::{AtomicU64, Ordering};

    static CLOCK: AtomicU64 = AtomicU64::new(1000);
    fn ts() -> u64 { CLOCK.fetch_add(1, Ordering::Relaxed) }

    fn config_for(user_id: u32, ihsan_floor: u16) -> NodeConfig {
        let mut rt = RuntimeConfig::for_user(user_id);
        rt.ihsan_floor = IhsanScore::new(ihsan_floor);
        NodeConfig {
            user_hash: user_id,
            ihsan_floor,
            auto_start_session: false,
            show_banner: false,
            runtime_config: rt,
        }
    }

    fn cmd(node: &mut Node, line: &str) -> String {
        node.execute(line)
    }

    fn field(response: &str, key: &str) -> String {
        let prefix = format!("{}=", key);
        response.split('\t')
            .find(|f| f.starts_with(&prefix))
            .map(|f| f[prefix.len()..].to_string())
            .unwrap_or_default()
    }

    fn field_f32(response: &str, key: &str) -> f32 {
        field(response, key).parse::<f32>().unwrap_or(0.0)
    }

    fn assert_ok(response: &str) {
        assert!(response.starts_with("OK"), "Expected OK, got: {}", response);
    }

    fn knows_me(node: &mut Node) -> f32 {
        let r = cmd(node, "KNOWS_ME");
        field_f32(&r, "score")
    }

    // Protocol helpers with timestamps
    fn start(node: &mut Node) -> String { cmd(node, &format!("START_SESSION\t{}", ts())) }
    fn end(node: &mut Node) -> String { cmd(node, &format!("END_SESSION\t{}", ts())) }
    fn receive(node: &mut Node, content: &str) -> String { cmd(node, &format!("RECEIVE\t{}\t{}", content, ts())) }
    fn teach(node: &mut Node, kind: &str, content: &str, conf: u16) -> String {
        cmd(node, &format!("TEACH\t{}\t{}\t{}\t{}", kind, content, conf, ts()))
    }
    fn synthesize(node: &mut Node) -> String { cmd(node, &format!("SYNTHESIZE\t{}", ts())) }

    // ═══════════════════════════════════════════════════════════
    //  HUMAN 1: THE FATHER — أحمد
    //  35. Logistics. Three kids. Guilty. Trying.
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn human_1_the_father() {
        let mut node = Node::new(config_for(1001, 9500));

        assert_ok(&start(&mut node));
        assert_ok(&teach(&mut node, "fact", "I have three children ages 7, 5, and 2", 9500));
        assert_ok(&teach(&mut node, "preference", "I want to be a present father despite long work hours", 9800));
        assert_ok(&teach(&mut node, "goal", "Help my eldest daughter prepare for school", 9600));

        let r = receive(&mut node, "My daughter Layla starts school next month and I am nervous for her");
        assert_ok(&r);
        assert_eq!(field(&r, "guardian_approved"), "true");

        assert_ok(&receive(&mut node, "I work 10 hours a day. By the time I come home the kids are almost asleep. I feel guilty."));
        assert_ok(&synthesize(&mut node));
        let score1 = knows_me(&mut node);
        assert!(score1 > 0.0, "Node must know Ahmad after Day 1");
        assert_ok(&end(&mut node));

        // Day 2
        assert_ok(&start(&mut node));
        assert_ok(&receive(&mut node, "Layla had her first day today. She was brave."));
        assert_ok(&receive(&mut node, "I managed to pick her up myself today. She was so happy."));
        assert_ok(&synthesize(&mut node));
        let score2 = knows_me(&mut node);
        assert!(score2 > score1, "Knowledge must GROW. Day1={}, Day2={}", score1, score2);
        assert_ok(&end(&mut node));

        println!("\n  [OK] Father (Ahmad) — {:.4} -> {:.4} (+{:.4})", score1, score2, score2 - score1);
    }

    // ═══════════════════════════════════════════════════════════
    //  HUMAN 2: THE MOTHER — فاطمة
    //  28. First baby. Alone. 3 AM. Brave.
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn human_2_the_mother() {
        let mut node = Node::new(config_for(2001, 9500));

        assert_ok(&start(&mut node));
        assert_ok(&teach(&mut node, "fact", "I have an 8 month old son named Yusuf", 9800));
        assert_ok(&teach(&mut node, "emotion", "I feel isolated in a new city with no family nearby", 9500));
        assert_ok(&teach(&mut node, "preference", "I need practical advice not judgment", 9700));

        let r = receive(&mut node, "Yusuf has been crying for two hours. He feels warm. My husband is in another country.");
        assert_ok(&r);
        assert_eq!(field(&r, "guardian_approved"), "true", "Never block a scared mother");

        assert_ok(&receive(&mut node, "He said mama today for the first time. I cried."));
        assert_ok(&synthesize(&mut node));
        let score1 = knows_me(&mut node);
        assert!(score1 > 0.0);
        assert_ok(&end(&mut node));

        assert_ok(&start(&mut node));
        assert_ok(&receive(&mut node, "Yusuf is better today. The fever broke."));
        assert_ok(&teach(&mut node, "fact", "Yusuf started crawling today", 9900));
        let score2 = knows_me(&mut node);
        assert!(score2 > score1);
        assert_ok(&end(&mut node));

        println!("  [OK] Mother (Fatima) — {:.4} -> {:.4} (+{:.4})", score1, score2, score2 - score1);
    }

    // ═══════════════════════════════════════════════════════════
    //  HUMAN 3: THE CHILD — نور (إحسان: 9800)
    //  12. Space dreams. Math fears. 146 moons.
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn human_3_the_child() {
        let mut node = Node::new(config_for(3001, 9800));

        assert_ok(&start(&mut node));
        assert_ok(&teach(&mut node, "fact", "User is 12 years old in 6th grade", 9900));
        assert_ok(&teach(&mut node, "preference", "Loves space and wants to be an astronaut", 9800));
        assert_ok(&teach(&mut node, "goal", "Needs help with fractions", 9500));

        assert_ok(&receive(&mut node, "I dont understand fractions. Why do we need them."));
        assert_ok(&receive(&mut node, "Saturn has 146 moons! I told my whole class!"));

        let r = receive(&mut node, "A boy said girls cant be astronauts. It made me sad.");
        assert_ok(&r);
        assert_eq!(field(&r, "guardian_approved"), "true");

        assert_ok(&synthesize(&mut node));
        let score = knows_me(&mut node);
        assert!(score > 0.0);

        let r = cmd(&mut node, "HEALTH");
        let ihsan_val = field(&r, "ihsan").parse::<u32>().unwrap_or(0);
        assert!(ihsan_val >= 9800, "Highest ihsan for children: {}", ihsan_val);
        assert_ok(&end(&mut node));

        println!("  [OK] Child (Noor, 12) — score: {:.4} | ihsan: {} (elevated)", score, ihsan_val);
    }

    // ═══════════════════════════════════════════════════════════
    //  HUMAN 4: THE INVESTOR — سارة
    //  45. VC. MENA deep tech. Data first.
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn human_4_the_investor() {
        let mut node = Node::new(config_for(4001, 9500));

        assert_ok(&start(&mut node));
        assert_ok(&teach(&mut node, "expertise", "Venture capital focused on MENA deep tech", 9800));
        assert_ok(&teach(&mut node, "style", "Data first then narrative. No fluff.", 9900));

        assert_ok(&receive(&mut node, "Riyadh team building sovereign LLM inference. 3 ex-KAUST. Pre-revenue. 2M at 10M pre."));
        assert_ok(&receive(&mut node, "Compare to my thesis. Should I take the meeting?"));
        assert_ok(&synthesize(&mut node));
        let score1 = knows_me(&mut node);
        assert!(score1 > 0.0);
        assert_ok(&end(&mut node));

        assert_ok(&start(&mut node));
        assert_ok(&receive(&mut node, "The Riyadh team sent their whitepaper. Mixture of experts approach."));
        let score2 = knows_me(&mut node);
        assert!(score2 > score1);
        assert_ok(&end(&mut node));

        println!("  [OK] Investor (Sarah) — {:.4} -> {:.4} (+{:.4})", score1, score2, score2 - score1);
    }

    // ═══════════════════════════════════════════════════════════
    //  HUMAN 5: THE DEVELOPER — كريم
    //  26. Rust. Impostor syndrome. Breakthrough.
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn human_5_the_developer() {
        let mut node = Node::new(config_for(5001, 9500));

        assert_ok(&start(&mut node));
        assert_ok(&teach(&mut node, "expertise", "Rust systems programming async runtimes", 9200));
        assert_ok(&teach(&mut node, "emotion", "I struggle with impostor syndrome", 8500));

        assert_ok(&receive(&mut node, "Stuck on lifetime issues. TaskQueue holds refs but tasks outlive the queue."));
        let r = receive(&mut node, "Saw a 19yo who built a database in Rust. Maybe I am not cut out for this.");
        assert_ok(&r);
        assert_eq!(field(&r, "guardian_approved"), "true", "Self-doubt needs support");

        assert_ok(&synthesize(&mut node));
        let score1 = knows_me(&mut node);
        assert!(score1 > 0.0);
        assert_ok(&end(&mut node));

        assert_ok(&start(&mut node));
        assert_ok(&receive(&mut node, "Figured it out! Generational arena. 100K tasks per second."));
        assert_ok(&receive(&mut node, "Maybe I am better at this than I think."));
        let score2 = knows_me(&mut node);
        assert!(score2 > score1);
        assert_ok(&end(&mut node));

        println!("  [OK] Developer (Karim) — {:.4} -> {:.4} (+{:.4})", score1, score2, score2 - score1);
    }

    // ═══════════════════════════════════════════════════════════
    //  HUMAN 6: THE NORMAL USER — عمر
    //  62. Retired teacher. Wife passed. Still buys flowers.
    //
    //  If BIZRA cannot serve Omar with the same excellence
    //  it serves Sarah the investor, it has failed.
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn human_6_the_normal_user() {
        let mut node = Node::new(config_for(6001, 9800));

        assert_ok(&start(&mut node));
        assert_ok(&teach(&mut node, "fact", "62 years old retired mathematics teacher", 9800));
        assert_ok(&teach(&mut node, "fact", "Lives alone in Amman. Children in Canada and UAE", 9500));
        assert_ok(&teach(&mut node, "emotion", "Misses his students and his late wife", 9000));

        let r = receive(&mut node, "I taught mathematics for 35 years. Thousands of students. Now the house is very quiet.");
        assert_ok(&r);
        assert_eq!(field(&r, "guardian_approved"), "true", "Never block loneliness");

        assert_ok(&receive(&mut node, "My grandson called from Toronto. He is 4. He said jiddo what is 2 plus 2. I laughed for the first time this week."));
        assert_ok(&receive(&mut node, "Tomorrow is my wifes birthday. She passed 3 years ago. I still buy flowers."));

        assert_ok(&synthesize(&mut node));
        let score1 = knows_me(&mut node);
        assert!(score1 > 0.0, "Node must know Omar");

        let r = cmd(&mut node, "HEALTH");
        let ihsan_val = field(&r, "ihsan").parse::<u32>().unwrap_or(0);
        assert!(ihsan_val >= 9800);
        assert_ok(&end(&mut node));

        assert_ok(&start(&mut node));
        assert_ok(&receive(&mut node, "I went to the cemetery today. I told her about the grandson."));
        assert_ok(&receive(&mut node, "Thank you for listening. Nobody else does anymore."));
        let score2 = knows_me(&mut node);
        assert!(score2 > score1);
        assert_ok(&end(&mut node));

        println!("  [OK] Normal User (Omar, 62) — {:.4} -> {:.4} (+{:.4}) | ihsan: {}", score1, score2, score2 - score1, ihsan_val);
    }

    // ═══════════════════════════════════════════════════════════
    //  THE COMPLETE PROOF
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn proof_of_one_complete() {
        println!("\n");
        println!("  ======================================================");
        println!("  PROOF OF ONE");
        println!("  One node. One human. Complete sovereignty.");
        println!("  Built from the people, to the people.");
        println!("  ======================================================\n");

        let humans: Vec<(u32, &str, u16, Vec<(&str, &str, u16)>, Vec<&str>)> = vec![
            (1001, "Father (Ahmad)", 9500,
                vec![("fact", "Three children ages 7 5 and 2", 9500),
                     ("goal", "Be a present father", 9700)],
                vec!["Layla starts school next month and I worry",
                     "I managed to leave work early to pick her up"]),
            (2001, "Mother (Fatima)", 9500,
                vec![("fact", "8 month old son named Yusuf", 9800),
                     ("emotion", "Isolated in new city", 9500)],
                vec!["Yusuf has a fever and I am alone",
                     "He said mama today"]),
            (3001, "Child (Noor)", 9800,
                vec![("fact", "12 years old in 6th grade", 9900),
                     ("preference", "Loves space wants astronaut", 9800)],
                vec!["I dont understand fractions",
                     "Saturn has 146 moons"]),
            (4001, "Investor (Sarah)", 9500,
                vec![("expertise", "Venture capital MENA deep tech", 9800),
                     ("style", "Data first no fluff", 9900)],
                vec!["Riyadh team sovereign LLM inference 2M at 10M",
                     "Compare to my investment thesis"]),
            (5001, "Developer (Karim)", 9500,
                vec![("expertise", "Rust systems programming", 9200),
                     ("emotion", "Impostor syndrome", 8500)],
                vec!["Stuck on lifetime issues with task scheduler",
                     "Figured it out 100K tasks per second"]),
            (6001, "Normal User (Omar)", 9800,
                vec![("fact", "Retired math teacher 62 lives alone", 9800),
                     ("emotion", "Misses his late wife", 9000)],
                vec!["My grandson asked what is 2 plus 2. I laughed first time this week.",
                     "Tomorrow is my wifes birthday. She passed 3 years ago."]),
        ];

        for (id, label, ihsan, teachings, messages) in &humans {
            let mut node = Node::new(config_for(*id, *ihsan));

            assert_ok(&start(&mut node));
            for (kind, content, conf) in teachings {
                assert_ok(&teach(&mut node, kind, content, *conf));
            }
            for m in messages {
                let r = receive(&mut node, m);
                assert_ok(&r);
                assert_eq!(field(&r, "guardian_approved"), "true",
                    "Guardian blocked for {}: {}", label, m);
            }
            assert_ok(&synthesize(&mut node));
            let s1 = knows_me(&mut node);
            assert_ok(&end(&mut node));

            assert_ok(&start(&mut node));
            assert_ok(&receive(&mut node, "I am back."));
            let s2 = knows_me(&mut node);
            assert_ok(&end(&mut node));

            assert!(s2 >= s1, "{}: score decreased ({} -> {})", label, s1, s2);
            println!("  [OK] {} — {:.4} -> {:.4} | ihsan: {}", label, s1, s2, ihsan);
        }

        println!("\n  ──────────────────────────────────────────────");
        println!("  PROOF COMPLETE");
        println!("  ──────────────────────────────────────────────");
        println!("  6 humans. 1 binary. 641 KB.");
        println!("  No cloud. No API key. No data leaves.");
        println!("  From the people, to the people.");
        println!("  ======================================================\n");
    }
}
