import React from 'react';
import {
  LayoutDashboard,
  Cpu,
  FlaskConical,
  Activity,
  BarChart3,
  GitCompare,
  Database,
  Network,
  BookOpen,
  Users,
  ShieldAlert,
} from 'lucide-react';

export type PageId =
  | 'overview'
  | 'hidden-state'
  | 'playground'
  | 'trajectory'
  | 'results'
  | 'comparison'
  | 'datasets'
  | 'methodology'
  | 'docs'
  | 'team';

interface SidebarProps {
  currentPage: PageId;
  onSelectPage: (page: PageId) => void;
}

const navItems: Array<{ id: PageId; label: string; icon: React.ElementType; badge?: string }> = [
  { id: 'overview', label: 'Research Overview', icon: LayoutDashboard },
  { id: 'hidden-state', label: 'Hidden-State Explorer', icon: Cpu, badge: 'Flagship' },
  { id: 'playground', label: 'Detection Playground', icon: FlaskConical },
  { id: 'trajectory', label: 'Trajectory Analysis', icon: Activity },
  { id: 'results', label: 'Experiment Results', icon: BarChart3 },
  { id: 'comparison', label: 'Model Comparison', icon: GitCompare },
  { id: 'datasets', label: 'Dataset Explorer', icon: Database },
  { id: 'methodology', label: 'System Architecture', icon: Network },
  { id: 'docs', label: 'Documentation Portal', icon: BookOpen },
  { id: 'team', label: 'Team & Attribution', icon: Users },
];

export const Sidebar: React.FC<SidebarProps> = ({ currentPage, onSelectPage }) => {
  return (
    <aside className="w-64 bg-slate-950 border-r border-slate-800/80 flex flex-col shrink-0 h-screen sticky top-0">
      {/* Brand & Group Tag */}
      <div className="p-5 border-b border-slate-800/80 flex items-center space-x-3">
        <div className="h-10 w-10 rounded-xl bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400 font-mono font-bold text-sm shadow-inner">
          G76
        </div>
        <div>
          <div className="font-bold text-white text-sm tracking-tight flex items-center space-x-1.5">
            <span>RTHD Platform</span>
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-blue-950 text-blue-300 border border-blue-800/60">
              v1.0
            </span>
          </div>
          <p className="text-[11px] text-slate-400 font-sans">Medical LLM Observability</p>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
        <div className="text-[10px] font-mono uppercase tracking-wider text-slate-500 px-3 mb-2 font-semibold">
          Platform Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectPage(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                isActive
                  ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30 font-semibold'
                  : 'text-slate-300 hover:bg-slate-900 hover:text-white border border-transparent'
              }`}
            >
              <div className="flex items-center space-x-2.5">
                <Icon className={`h-4 w-4 ${isActive ? 'text-blue-400' : 'text-slate-400'}`} />
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300 border border-blue-500/40 font-bold uppercase">
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Clinical Disclaimer Box */}
      <div className="p-3 border-t border-slate-800/80">
        <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800 text-[11px] text-slate-400 space-y-1">
          <div className="flex items-center space-x-1.5 text-amber-400 font-semibold">
            <ShieldAlert className="h-3.5 w-3.5 shrink-0" />
            <span>Research Tool Only</span>
          </div>
          <p className="text-[10px] text-slate-400 leading-snug">
            Evaluated on 4 pilot benchmarks. Not a certified clinical medical device.
          </p>
        </div>
      </div>
    </aside>
  );
};
