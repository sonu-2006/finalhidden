import React, { useState } from 'react';
import { Users, Award, Copy, Check, ExternalLink, GraduationCap, GitBranch } from 'lucide-react';

export const TeamInfo: React.FC = () => {
  const [copied, setCopied] = useState(false);

  const bibtex = `@article{rthd_group76_2026,
  title={Medical Hallucination Detection in Large Language Models Using Hidden-State Probing},
  author={Group 76 and Dr. Swapna T R},
  journal={Department of Computer Science & Engineering, Final Year Capstone Project},
  year={2026},
  institution={Department of Computer Science & Engineering},
  note={Group ID: G076. Source Repository: https://github.com/sonu-2006/finalhidden}
}`;

  const handleCopy = () => {
    navigator.clipboard.writeText(bibtex);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-8 p-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-800 pb-4">
        <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
          <Users className="h-6 w-6 text-blue-400" />
          <span>Research Team & Academic Attribution</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Final Year B.Tech Project Defense &bull; Group 76 &bull; Department of Computer Science & Engineering
        </p>
      </div>

      {/* Team Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-3">
          <div className="flex items-center space-x-2 text-xs font-mono text-blue-400 uppercase font-bold">
            <GraduationCap className="h-4 w-4" />
            <span>Project Team</span>
          </div>
          <h3 className="text-xl font-bold text-white tracking-tight">Group 76</h3>
          <p className="text-xs text-slate-400 font-mono">Department of Computer Science & Engineering</p>
          <div className="pt-2 text-xs text-slate-300 leading-relaxed space-y-1">
            <p>
              • Conducted 4-bit model deployment and GPU memory optimization on Kaggle.
            </p>
            <p>
              • Implemented hidden-state feature caching across layers 4, 20, and 32.
            </p>
            <p>
              • Executed probe training, critical lead-time evaluation, and ML observability platform development.
            </p>
          </div>
        </div>

        <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-3">
          <div className="flex items-center space-x-2 text-xs font-mono text-emerald-400 uppercase font-bold">
            <Award className="h-4 w-4" />
            <span>Faculty Advisor & Guide</span>
          </div>
          <h3 className="text-xl font-bold text-emerald-300 tracking-tight">Dr. Swapna T R</h3>
          <p className="text-xs text-slate-400 font-mono">Associate Professor, Dept. of Computer Science & Engineering</p>
          <div className="pt-2 text-xs text-slate-300 leading-relaxed space-y-1">
            <p>
              • Academic capstone project supervision, research problem validation, and evaluation rubric coordination.
            </p>
            <p>
              • Methodological alignment with clinical benchmark datasets and ethical governance review.
            </p>
          </div>
        </div>
      </div>

      {/* Project Metadata & Repository Links */}
      <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
        <h2 className="text-sm font-bold text-white">Project Metadata & Repository Links</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-400 block text-[10px]">Academic Group ID</span>
            <span className="text-blue-400 font-bold text-sm">Group 76 (G076)</span>
          </div>
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-400 block text-[10px]">GitHub Repository</span>
            <a
              href="https://github.com/sonu-2006/finalhidden"
              target="_blank"
              rel="noreferrer"
              className="text-white font-bold text-xs hover:text-blue-400 flex items-center space-x-1 mt-0.5"
            >
              <span>sonu-2006/finalhidden</span>
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-400 block text-[10px]">Status</span>
            <span className="text-emerald-400 font-bold text-sm">Ready for Faculty Defense</span>
          </div>
        </div>
      </div>

      {/* BibTeX Citation Box */}
      <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono uppercase text-slate-400 font-bold">BibTeX Project Citation</span>
          <button
            onClick={handleCopy}
            className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-200 text-xs font-mono border border-slate-700 flex items-center space-x-1.5 transition"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            <span>{copied ? 'Copied' : 'Copy BibTeX'}</span>
          </button>
        </div>
        <pre className="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 overflow-x-auto leading-relaxed">
          <code>{bibtex}</code>
        </pre>
      </div>
    </div>
  );
};
