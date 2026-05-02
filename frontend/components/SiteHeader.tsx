import Link from "next/link";
import { History, LayoutDashboard, Sparkles } from "lucide-react";

export function SiteHeader() {
  return (
    <header className="site-header">
      <div className="site-header-inner page-width">
        <Link href="/" className="brand-link">
          <Sparkles size={18} />
          <span>StudyGuide Agent</span>
        </Link>

        <nav className="nav-links">
          <Link href="/dashboard" className="nav-pill">
            <LayoutDashboard size={14} />
            Dashboard
          </Link>
          <Link href="/history" className="nav-pill">
            <History size={14} />
            History
          </Link>
        </nav>
      </div>
    </header>
  );
}
