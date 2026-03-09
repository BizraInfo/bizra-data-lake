import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import OnboardingFlow from "./onboarding/OnboardingFlow";
import PersonaDashboard from "./bizra-dashboard";
import OperatorDashboard from "./node0-dashboard";
import ConstitutionalSeedPage from "./ConstitutionalSeedPage";
import { useNode } from "./useNode";

const GENESIS_EMAIL_KEY = "bizra.genesis.email";

function getHashParams() {
  const hash = window.location.hash || "";
  const queryIndex = hash.indexOf("?");
  return new URLSearchParams(queryIndex === -1 ? "" : hash.slice(queryIndex + 1));
}

function readGenesisEmail() {
  const emailFromHash = getHashParams().get("email");

  if (emailFromHash) {
    try {
      window.localStorage.setItem(GENESIS_EMAIL_KEY, emailFromHash);
    } catch {}
    return emailFromHash;
  }

  try {
    return window.localStorage.getItem(GENESIS_EMAIL_KEY) || "";
  } catch {
    return "";
  }
}

function isLandingRequest() {
  const hash = window.location.hash || "";
  return (
    hash === "" ||
    hash === "#" ||
    hash.startsWith("#/landing") ||
    hash.startsWith("#/showcase") ||
    hash.startsWith("#/product") ||
    hash.startsWith("#/demo")
  );
}

function getSurfaceFromHash() {
  const hash = window.location.hash || "";

  if (hash.startsWith("#/onboarding")) {
    return "onboarding";
  }

  if (hash.startsWith("#/dashboard")) {
    return "dashboard";
  }

  if (hash.startsWith("#/node0") || hash.startsWith("#/ops") || hash.startsWith("#/operator")) {
    return "operator";
  }

  if (hash.startsWith("#/seed") || hash.startsWith("#/constitution") || hash.startsWith("#/block-0")) {
    return "seed";
  }

  if (hash.startsWith("#/app") || hash.startsWith("#/chat")) {
    return "app";
  }

  return "redirect";
}

function OnboardingSurface() {
  const node = useNode();
  return (
    <OnboardingFlow
      node={node}
      initialEmail={readGenesisEmail()}
      onComplete={() => { window.location.hash = "#/dashboard"; }}
    />
  );
}

function Root() {
  const [surface, setSurface] = useState(getSurfaceFromHash);

  useEffect(() => {
    if (isLandingRequest()) {
      window.location.replace("./bizra-flagship.html");
      return undefined;
    }

    const onHashChange = () => {
      if (isLandingRequest()) {
        window.location.replace("./bizra-flagship.html");
        return;
      }
      setSurface(getSurfaceFromHash());
    };

    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  if (surface === "onboarding") {
    return <OnboardingSurface />;
  }

  if (surface === "dashboard") {
    return <PersonaDashboard />;
  }

  if (surface === "operator") {
    return <OperatorDashboard />;
  }

  if (surface === "seed") {
    return <ConstitutionalSeedPage />;
  }

  if (surface === "app") {
    return <App />;
  }

  return null;
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
