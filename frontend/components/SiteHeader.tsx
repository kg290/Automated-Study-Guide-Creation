import Link from "next/link";
import { Sparkles } from "lucide-react";

export function SiteHeader() {
  return (
    <header className="site-header">
      <div className="site-header-inner page-width">
        <Link href="/" className="brand-link">
          <Sparkles size={18} />
          <span>StudyGuide Agent</span>
        </Link>

        <nav className="nav-links">
          <Link href="/dashboard">Dashboard</Link>
          <Link href="/history">History</Link>
        </nav>
      </div>
    </header>
  );
}
