import React, { useState } from 'react';
import { Sidebar, PageId } from './components/Sidebar';
import { Navbar } from './components/Navbar';
import { ResearchOverview } from './pages/ResearchOverview';
import { HiddenStateExplorer } from './pages/HiddenStateExplorer';
import { DetectionPlayground } from './pages/DetectionPlayground';
import { TrajectoryAnalysis } from './pages/TrajectoryAnalysis';
import { ExperimentResults } from './pages/ExperimentResults';
import { ModelComparison } from './pages/ModelComparison';
import { DatasetExplorer } from './pages/DatasetExplorer';
import { Methodology } from './pages/Methodology';
import { Documentation } from './pages/Documentation';
import { TeamInfo } from './pages/TeamInfo';

export function App() {
  const [currentPage, setCurrentPage] = useState<PageId>('overview');

  const renderPage = () => {
    switch (currentPage) {
      case 'overview':
        return <ResearchOverview onNavigate={(p) => setCurrentPage(p)} />;
      case 'hidden-state':
        return <HiddenStateExplorer />;
      case 'playground':
        return <DetectionPlayground />;
      case 'trajectory':
        return <TrajectoryAnalysis />;
      case 'results':
        return <ExperimentResults />;
      case 'comparison':
        return <ModelComparison />;
      case 'datasets':
        return <DatasetExplorer />;
      case 'methodology':
        return <Methodology />;
      case 'docs':
        return <Documentation />;
      case 'team':
        return <TeamInfo />;
      default:
        return <ResearchOverview onNavigate={(p) => setCurrentPage(p)} />;
    }
  };

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-100">
      {/* Fixed Sidebar */}
      <Sidebar currentPage={currentPage} onSelectPage={(p) => setCurrentPage(p)} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Navbar currentPage={currentPage} />
        <main className="flex-1 overflow-y-auto pb-12">{renderPage()}</main>
      </div>
    </div>
  );
}

export default App;
