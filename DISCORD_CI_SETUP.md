# 🔔 Discord CI/CD Notifications Setup

**Goal:** Send real-time CI/CD updates to Discord so the community can see progress  
**Time:** 15 minutes  
**Skill Level:** Beginner-friendly

---

## 🎯 What You'll Get

After this setup, your Discord #dev channel will receive automated messages like:

```
✅ CI Pipeline PASSED
  Repository: bizra-data-lake
  Branch: main
  Commit: c5d81ba
  Message: "feat(benchmark): add GEM modules"
  Duration: 14 minutes
  
  View Run: https://github.com/BizraInfo/bizra-data-lake/actions/runs/22420859045
```

---

## 📋 Prerequisites

- Discord server (admin access)
- GitHub repository (admin access)
- 15 minutes

---

## Step 1: Create Discord Webhook

### 1.1 Navigate to Server Settings

1. Open Discord
2. Right-click your server (e.g., "BIZRA Community")
3. Click **Server Settings**

### 1.2 Create Webhook

1. Click **Integrations** (left sidebar)
2. Click **Webhooks** tab
3. Click **New Webhook** button
4. Configure:
   - **Name:** "CI/CD Bot"
   - **Channel:** #dev (or #ci-cd)
   - **Avatar:** (optional) Upload a robot/gear icon

### 1.3 Copy Webhook URL

1. Click **Copy Webhook URL**
2. Save it somewhere safe (you'll need it in Step 2)

**⚠️ Security Warning:** This URL is like a password. Anyone with it can post to your channel. Never commit it to GitHub!

---

## Step 2: Add Secret to GitHub

### 2.1 Navigate to Repository Settings

1. Go to your GitHub repository
2. Click **Settings** (top menu)
3. Click **Secrets and variables** → **Actions** (left sidebar)

### 2.2 Create New Secret

1. Click **New repository secret**
2. Configure:
   - **Name:** `DISCORD_WEBHOOK`
   - **Secret:** (paste the webhook URL from Step 1.3)
3. Click **Add secret**

---

## Step 3: Update GitHub Actions Workflow

### 3.1 Open Your CI Workflow

Edit `.github/workflows/ci.yml` in your repository.

### 3.2 Add Discord Notification Job

Add this at the END of the file (after all other jobs):

```yaml
  # ═══════════════════════════════════════════════════════════════════════════
  # Discord Notification
  # ═══════════════════════════════════════════════════════════════════════════
  notify-discord:
    name: Notify Discord
    runs-on: ubuntu-latest
    needs: [lint-python, lint-rust, test-python, test-rust, test-pyo3]
    if: always()
    steps:
      - name: Send success notification
        if: ${{ needs.lint-python.result == 'success' && needs.lint-rust.result == 'success' && needs.test-python.result == 'success' && needs.test-rust.result == 'success' && needs.test-pyo3.result == 'success' }}
        uses: sarisia/actions-status-discord@v1
        with:
          webhook: ${{ secrets.DISCORD_WEBHOOK }}
          title: "✅ CI Pipeline PASSED"
          description: |
            **Repository:** ${{ github.repository }}
            **Branch:** ${{ github.ref_name }}
            **Commit:** ${{ github.sha }}
            **Message:** "${{ github.event.head_commit.message }}"
            
            [View Run](${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }})
          color: 0x00ff00
          username: "BIZRA CI Bot"
          avatar_url: "https://i.imgur.com/4M34hi2.png"

      - name: Send failure notification
        if: ${{ needs.lint-python.result == 'failure' || needs.lint-rust.result == 'failure' || needs.test-python.result == 'failure' || needs.test-rust.result == 'failure' || needs.test-pyo3.result == 'failure' }}
        uses: sarisia/actions-status-discord@v1
        with:
          webhook: ${{ secrets.DISCORD_WEBHOOK }}
          title: "❌ CI Pipeline FAILED"
          description: |
            **Repository:** ${{ github.repository }}
            **Branch:** ${{ github.ref_name }}
            **Commit:** ${{ github.sha }}
            **Message:** "${{ github.event.head_commit.message }}"
            
            **Failed Jobs:**
            - Lint Python: ${{ needs.lint-python.result }}
            - Lint Rust: ${{ needs.lint-rust.result }}
            - Test Python: ${{ needs.test-python.result }}
            - Test Rust: ${{ needs.test-rust.result }}
            - Test PyO3: ${{ needs.test-pyo3.result }}
            
            [View Run](${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }})
          color: 0xff0000
          username: "BIZRA CI Bot"
          avatar_url: "https://i.imgur.com/4M34hi2.png"
```

### 3.3 Adjust Job Dependencies

**Important:** The `needs:` line should list ALL your CI jobs. Update it to match your actual job names.

For example, if your workflow has these jobs:
- `lint-python`
- `lint-rust`
- `test-python`
- `test-rust`
- `build-frontend`

Then update the `needs:` line:
```yaml
needs: [lint-python, lint-rust, test-python, test-rust, build-frontend]
```

---

## Step 4: Test the Setup

### 4.1 Make a Test Commit

```bash
# Make a trivial change
echo "# Testing Discord notifications" >> README.md

# Commit and push
git add README.md
git commit -m "test: verify Discord CI notifications"
git push origin main
```

### 4.2 Watch Discord

1. Go to your Discord #dev channel
2. Wait 2-3 minutes
3. You should see a message from "BIZRA CI Bot"

### 4.3 Verify

If you see the notification: ✅ Success!

If you don't:
- Check GitHub Actions secrets (Step 2)
- Check webhook URL is correct
- Check Discord channel permissions
- Check the `needs:` job list matches your workflow

---

## 🎨 Customization Options

### Change Colors

```yaml
color: 0x00ff00  # Green (success)
color: 0xff0000  # Red (failure)
color: 0xffff00  # Yellow (warning)
color: 0x3498db  # Blue (info)
```

### Change Bot Name

```yaml
username: "BIZRA CI Bot"      # Default
username: "Genesis Guardian"   # Alternative
username: "Build Status"       # Simple
```

### Add More Details

```yaml
description: |
  **Repository:** ${{ github.repository }}
  **Branch:** ${{ github.ref_name }}
  **Author:** ${{ github.actor }}
  **Duration:** ${{ github.event.workflow_run.duration }}ms
  **Tests:** ${{ needs.test-python.outputs.test_count }} passed
  
  [View Run](${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }})
```

---

## 🚀 Advanced: Notification Filters

### Only Notify on Main Branch

Add this condition to the job:

```yaml
notify-discord:
  if: always() && github.ref == 'refs/heads/main'
```

### Only Notify on Failures

```yaml
notify-discord:
  if: failure()
```

### Notify on Both Success and Failure (Default)

```yaml
notify-discord:
  if: always()
```

---

## 📊 Multiple Channels

Want different notifications in different channels?

### Create Multiple Webhooks

1. Create webhook for #dev → `DISCORD_WEBHOOK_DEV`
2. Create webhook for #announcements → `DISCORD_WEBHOOK_ANNOUNCE`

### Use Different Webhooks for Different Events

```yaml
# In ci.yml - notify developers
webhook: ${{ secrets.DISCORD_WEBHOOK_DEV }}

# In release.yml - notify everyone
webhook: ${{ secrets.DISCORD_WEBHOOK_ANNOUNCE }}
```

---

## 🐛 Troubleshooting

### Problem: No notifications appearing

**Solution 1:** Check GitHub Actions logs
1. Go to Actions tab
2. Click the latest run
3. Check if `notify-discord` job ran
4. Check for error messages

**Solution 2:** Verify webhook URL
1. Copy webhook URL from Discord
2. Go to GitHub Settings → Secrets
3. Update `DISCORD_WEBHOOK` secret
4. Try again

### Problem: "Error: Invalid Webhook" in GitHub Actions

**Cause:** Webhook URL is incorrect or expired

**Solution:**
1. Go to Discord Server Settings → Integrations → Webhooks
2. Delete old webhook
3. Create new webhook
4. Copy new URL
5. Update GitHub secret

### Problem: Notifications work but formatting is broken

**Cause:** YAML formatting error

**Solution:**
1. Check indentation (use spaces, not tabs)
2. Validate YAML: https://www.yamllint.com/
3. Ensure multi-line strings use `|` properly

---

## 🎯 Next Steps

Once notifications work:

1. **Customize messages** - Add emojis, better formatting
2. **Add more events** - Deployments, releases, etc.
3. **Create dedicated channel** - #ci-cd for build updates
4. **Set up roles** - @dev-team mentions on failures

---

## 📚 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Discord Webhooks Guide](https://support.discord.com/hc/en-us/articles/228383668)
- [sarisia/actions-status-discord](https://github.com/sarisia/actions-status-discord)

---

## ✅ Verification Checklist

- [ ] Discord webhook created
- [ ] GitHub secret added (`DISCORD_WEBHOOK`)
- [ ] Workflow updated with notification job
- [ ] Test commit pushed
- [ ] Notification appeared in Discord
- [ ] Success notification works
- [ ] Failure notification works (optional: trigger by breaking a test)

---

**Need help?** Ask in #support or open an issue!

**Working perfectly?** Share a screenshot in #showcase! 📸
