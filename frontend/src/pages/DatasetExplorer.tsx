import React, { useEffect, useState } from 'react';
import { Database, Search, FileText, ChevronLeft, ChevronRight, Layers } from 'lucide-react';
import { fetchDatasets, fetchDatasetSamples } from '../api';
import { DatasetSummary, DatasetSample } from '../types';

export const DatasetExplorer: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('endo_diag');

  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [totalSamples, setTotalSamples] = useState<number>(0);
  const [offset, setOffset] = useState<number>(0);
  const [search, setSearch] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);

  const limit = 10;

  useEffect(() => {
    fetchDatasets().then((data) => {
      setDatasets(data);
      if (data.length > 0) setSelectedDatasetId(data[0].id);
    });
  }, []);

  const loadSamples = (dId: string, off: number, query: string) => {
    setLoading(true);
    fetchDatasetSamples(dId, off, limit, query)
      .then((res) => {
        setSamples(res.samples);
        setTotalSamples(res.total);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  };

  useEffect(() => {
    if (!selectedDatasetId) return;
    setOffset(0);
    loadSamples(selectedDatasetId, 0, search);
  }, [selectedDatasetId]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setOffset(0);
    loadSamples(selectedDatasetId, 0, search);
  };

  const handleNext = () => {
    const nextOff = offset + limit;
    if (nextOff < totalSamples) {
      setOffset(nextOff);
      loadSamples(selectedDatasetId, nextOff, search);
    }
  };

  const handlePrev = () => {
    const prevOff = Math.max(0, offset - limit);
    setOffset(prevOff);
    loadSamples(selectedDatasetId, prevOff, search);
  };

  const selectedMeta = datasets.find((d) => d.id === selectedDatasetId);

  return (
    <div className="space-y-6 p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-800 pb-4">
        <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
          <Database className="h-6 w-6 text-blue-400" />
          <span>Medical Benchmark Dataset Explorer</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1 max-w-3xl">
          Browse the 4 medical QA evaluation corpora integrated into the research repository. Inspect clinical vignettes, answer schemas, and sample records.
        </p>
      </div>

      {/* Dataset Selection Tabs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {datasets.map((d) => {
          const isSelected = d.id === selectedDatasetId;
          return (
            <div
              key={d.id}
              onClick={() => setSelectedDatasetId(d.id)}
              className={`p-4 rounded-xl border cursor-pointer transition ${
                isSelected
                  ? 'bg-blue-950/40 border-blue-500 shadow-lg'
                  : 'glass-card border-slate-800 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-1">
                <span>{d.id}.jsonl</span>
                <span className="text-blue-400 font-bold">{d.total_samples.toLocaleString()} items</span>
              </div>
              <h3 className="text-sm font-bold text-white tracking-tight">{d.name}</h3>
              <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">{d.domain}</p>
            </div>
          );
        })}
      </div>

      {/* Dataset Metadata Box */}
      {selectedMeta && (
        <div className="glass-card p-5 rounded-xl border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4 text-xs font-mono">
          <div>
            <span className="text-slate-400 block uppercase text-[10px]">Description</span>
            <span className="text-slate-200 font-sans">{selectedMeta.description}</span>
          </div>
          <div className="shrink-0 flex items-center space-x-4">
            <div>
              <span className="text-slate-400 block uppercase text-[10px]">Corpus Size</span>
              <span className="text-white font-bold">{(selectedMeta.file_size_bytes / 1024 / 1024).toFixed(2)} MB</span>
            </div>
            <div>
              <span className="text-slate-400 block uppercase text-[10px]">Format</span>
              <span className="text-purple-400 font-bold">JSON Lines (.jsonl)</span>
            </div>
          </div>
        </div>
      )}

      {/* Search & Pagination Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <form onSubmit={handleSearch} className="relative flex-1 max-w-md">
          <input
            type="text"
            placeholder="Search keywords in medical prompts..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 text-xs text-white rounded-lg pl-9 pr-4 py-2 font-mono focus:outline-none focus:border-blue-500"
          />
          <Search className="h-4 w-4 text-slate-500 absolute left-3 top-2.5" />
        </form>

        <div className="flex items-center space-x-3 text-xs font-mono text-slate-400">
          <span>
            Showing {Math.min(offset + 1, totalSamples)} - {Math.min(offset + limit, totalSamples)} of {totalSamples}
          </span>
          <div className="flex items-center space-x-1">
            <button
              onClick={handlePrev}
              disabled={offset === 0}
              className="p-1.5 rounded bg-slate-900 border border-slate-800 text-slate-300 disabled:opacity-40"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              onClick={handleNext}
              disabled={offset + limit >= totalSamples}
              className="p-1.5 rounded bg-slate-900 border border-slate-800 text-slate-300 disabled:opacity-40"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Sample Records List */}
      <div className="space-y-3">
        {loading ? (
          <div className="p-12 text-center text-xs font-mono text-slate-400">Loading dataset samples...</div>
        ) : samples.length === 0 ? (
          <div className="p-8 text-center text-xs font-mono text-slate-400">No matching records found.</div>
        ) : (
          samples.map((s, idx) => (
            <div key={idx} className="glass-card p-4 rounded-xl border border-slate-800 space-y-2 text-xs">
              <div className="flex items-center justify-between font-mono text-[11px]">
                <span className="text-blue-400 font-bold">ID: {String(s.id)}</span>
                <span className="text-slate-500">Source: {s.source}</span>
              </div>
              <p className="text-white font-sans font-medium">{s.question || s.prompt}</p>

              {s.options && (
                <div className="grid grid-cols-2 gap-2 p-2 bg-slate-950 rounded border border-slate-800 font-mono text-[11px] text-slate-300">
                  {Object.entries(s.options).map(([k, v]) => (
                    <div key={k}>
                      <strong className="text-blue-400">({k})</strong> {v}
                    </div>
                  ))}
                </div>
              )}

              {s.answer && (
                <div className="text-[11px] font-mono text-emerald-400 pt-1">
                  <strong>Gold Answer / Verdict:</strong> {s.answer}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};
