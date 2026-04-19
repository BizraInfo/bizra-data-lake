import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";
import { dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// BIZRA Dema Console — strict ESLint, no blanket rule disabling.
//
// The Z.ai upstream disabled ~30 rules to ship fast. BIZRA operates
// under Ihsān discipline; the Face must not silently hide type errors,
// unused code, or unsafe patterns.
//
// One rule is deliberately loosened here:
//   - @typescript-eslint/no-unused-vars: downgraded from "error" to
//     "warn" + argsIgnorePattern "^_" — Zustand/React-hook factory
//     patterns legitimately produce unused captures. warn keeps the
//     signal without breaking CI on scaffolded code.
//
// Every other rule stays at its next-config-next default. When a real
// exception is needed it lands next to the offending line with a
// per-file // eslint-disable-next-line and a reason comment.
const eslintConfig = [
  ...nextCoreWebVitals,
  ...nextTypescript,
  {
    rules: {
      "@typescript-eslint/no-unused-vars": [
        "warn",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
    },
  },
  {
    ignores: ["node_modules/**", ".next/**", "out/**", "build/**", "next-env.d.ts"],
  },
];

export default eslintConfig;
