import React from 'react';
import { Activity, Terminal, ExternalLink, GitBranch } from 'lucide-react';
import { PageId } from './Sidebar';

interface NavbarProps {
  currentPage: PageId;
  activeSample?: string;
}

const pageTitles: Record<PageId, string> = {
  overview: 'Research Overview & Executive Dashboard',
  'hidden-state': 'Hidden-State Representation & Vector Explorer',
  playground: 'Hallucination Detection & Intervention Playground',
  trajectory: 'Generation Trajectory & Lead-Time Analysis',
  results: 'Verified Pilot Experiment Results & Metrics',
  comparison: 'Cross-Architecture Model Benchmark Comparison',
  datasets: 'Medical Benchmark Dataset Explorer',
  methodology: 'System Architecture & Mathematical Formulation',
  docs: 'Academic Capstone Documentation Portal',
  team: 'Research Team, Guide & BibTeX Attribution',
};

export const Navbar: React.FC<NavbarProps> = ({ currentPage }) => {
  return (
    <header className="h-16 bg-slate-950/80 backdrop-blur-md border-b border-slate-800/80 px-6 flex items-center justify-between sticky top-0 z-40">
      <div className="flex items-center space-x-3">
        <h1 className="text-sm sm:text-base font-bold text-white tracking-tight">
          {pageTitles[currentPage]}
        </h1>
        <span className="hidden md:inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-mono bg-slate-900 border border-slate-800 text-slate-400">
          <Terminal className="h-3 w-3 text-blue-400" />
          <span>Active Backend: FastAPI (127.0.0.1:8000)</span>
        </span>
      </div>

      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-2 px-2.5 py-1 rounded-full bg-emerald-950/40 border border-emerald-800/60 text-emerald-400 text-xs">
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span className="font-mono text-[11px] font-medium">Probes Loaded (4/4)</span>
        </div>

        <a
          href="https://github.com/sonu-2006/finalhidden"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-200 text-xs font-medium border border-slate-800 transition"
        >
          <GitBranch className="h-3.5 w-3.5" />
          <span>Repository</span>
          <ExternalLink className="h-3 w-3 text-slate-400" />
        </a>
      </div>
    </header>
  );
};
