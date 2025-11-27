import React, { useState, useEffect, useRef } from 'react';
import { Upload, Play, Square, Download, Activity, CheckCircle, AlertCircle, Video, Settings } from 'lucide-react';

const API_URL = 'http://localhost:8000'; // Change if deployed

export default function App() {
  const [file, setFile] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [stats, setStats] = useState({ shots: 0, baskets: 0, accuracy: 0 });
  
  // Settings
  const [testMode, setTestMode] = useState(false);
  const [processingMode, setProcessingMode] = useState('full_tracking');
  
  const [errorMsg, setErrorMsg] = useState('');
  const statusInterval = useRef(null);

  useEffect(() => {
    if (status === 'processing') {
      statusInterval.current = setInterval(checkStatus, 1000);
    } else {
      clearInterval(statusInterval.current);
    }
    return () => clearInterval(statusInterval.current);
  }, [status]);

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
      setStatus('idle');
      setProgress(0);
      setStats({ shots: 0, baskets: 0, accuracy: 0 });
    }
  };

  const uploadAndStart = async () => {
    if (!file) return;

    try {
      setStatus('uploading');
      const formData = new FormData();
      formData.append('file', file);

      const uploadRes = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
      if (!uploadRes.ok) throw new Error("Upload failed");
      const uploadData = await uploadRes.json();
      const fid = uploadData.file_id;
      setFileId(fid);

      // Include processing_mode in query
      const processRes = await fetch(
        `${API_URL}/process/${fid}?test_mode=${testMode}&mode=${processingMode}`, 
        { method: 'POST' }
      );
      
      if (!processRes.ok) throw new Error("Processing start failed");
      setStatus('processing');

    } catch (err) {
      console.error(err);
      setStatus('error');
      setErrorMsg(err.message);
    }
  };

  const checkStatus = async () => {
    if (!fileId) return;
    try {
      const res = await fetch(`${API_URL}/status/${fileId}`);
      const data = await res.json();

      if (data.status === 'processing') {
        setProgress(data.percentage);
        setStats(data.stats);
      } else if (data.status === 'completed') {
        setStatus('completed');
        setProgress(100);
        setStats(data.stats);
      } else if (data.status === 'error') {
        setStatus('error');
        setErrorMsg(data.message);
      } else if (data.status === 'stopped') {
        setStatus('stopped');
      }
    } catch (err) {
      console.error("Status check failed", err);
    }
  };

  const stopProcessing = async () => {
    if (!fileId) return;
    try {
      await fetch(`${API_URL}/stop/${fileId}`, { method: 'POST' });
      setStatus('stopped');
    } catch (err) {
      console.error(err);
    }
  };

  const downloadVideo = () => {
    window.open(`${API_URL}/download/${fileId}`, '_blank');
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white font-sans p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="flex flex-col items-center space-y-4">
          <div className="p-1 bg-gradient-to-r from-orange-500 to-yellow-400 rounded-2xl shadow-lg shadow-orange-500/20">
            <img 
              src="logo.png" 
              alt="SWISH AI Logo" 
              className="w-32 h-32 rounded-xl object-cover"
            />
          </div>
          <div className="text-center space-y-2">
            <h1 className="text-5xl font-extrabold bg-gradient-to-r from-orange-500 to-yellow-400 bg-clip-text text-transparent tracking-tight">
              SWISH AI
            </h1>
            <p className="text-slate-400 text-lg">Analyze your shots with computer vision</p>
          </div>
        </div>

        {/* Main Card */}
        <div className="bg-slate-800 rounded-2xl p-8 shadow-xl border border-slate-700">
          
          {/* Upload Section */}
          <div className="flex flex-col items-center justify-center border-2 border-dashed border-slate-600 rounded-xl p-8 transition-colors hover:border-orange-500/50">
            <input 
              type="file" 
              accept="video/*" 
              onChange={handleFileChange} 
              className="hidden" 
              id="video-upload"
              disabled={status === 'processing'}
            />
            <label htmlFor="video-upload" className="cursor-pointer flex flex-col items-center gap-4">
              <div className="w-16 h-16 bg-slate-700 rounded-full flex items-center justify-center text-orange-400">
                {file ? <Video size={32} /> : <Upload size={32} />}
              </div>
              <div className="text-center">
                <p className="text-lg font-medium">{file ? file.name : "Select Video File"}</p>
                <p className="text-sm text-slate-500 mt-1">MP4, MOV, AVI supported</p>
              </div>
            </label>
          </div>

          {/* Controls */}
          {file && (
            <div className="mt-8 flex flex-col gap-6">
              
              {/* Settings Row */}
              <div className="flex flex-wrap items-center justify-center gap-6 bg-slate-900/50 p-4 rounded-xl border border-slate-700">
                
                {/* Mode Selector */}
                <div className="flex flex-col gap-2">
                  <label className="text-xs text-slate-400 font-bold uppercase tracking-wider flex items-center gap-2">
                    <Settings size={14} /> Processing Mode
                  </label>
                  <select
                    value={processingMode}
                    onChange={(e) => setProcessingMode(e.target.value)}
                    disabled={status === 'processing'}
                    className="bg-slate-800 text-white border border-slate-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-orange-500"
                  >
                    <option value="full_tracking">Full Tracking (Boxes + Effects)</option>
                    <option value="stats_effects">Stats & Effects (Clean)</option>
                    <option value="stats_only">Stats Only (Minimal)</option>
                  </select>
                </div>

                {/* Test Mode Toggle */}
                <div className="flex items-center gap-3 pt-6">
                  <label className="flex items-center gap-2 cursor-pointer group">
                    <input 
                      type="checkbox" 
                      checked={testMode} 
                      onChange={(e) => setTestMode(e.target.checked)}
                      disabled={status === 'processing'}
                      className="w-5 h-5 accent-orange-500 rounded focus:ring-orange-500 focus:ring-2"
                    />
                    <span className="text-sm group-hover:text-white transition-colors">Test Mode (15 sec)</span>
                  </label>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex justify-center">
                {status === 'processing' ? (
                  <button 
                    onClick={stopProcessing}
                    className="flex items-center gap-2 bg-red-500 hover:bg-red-600 text-white px-8 py-3 rounded-xl font-bold transition-all shadow-lg shadow-red-500/20"
                  >
                    <Square size={20} /> Stop Analysis
                  </button>
                ) : (
                  <button 
                    onClick={uploadAndStart}
                    disabled={status === 'completed'}
                    className="flex items-center gap-2 bg-orange-500 hover:bg-orange-600 text-white px-8 py-3 rounded-xl font-bold transition-all shadow-lg shadow-orange-500/20 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105"
                  >
                    <Play size={20} /> {status === 'completed' ? 'Done' : 'Start Analysis'}
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Progress Section */}
          {status !== 'idle' && (
            <div className="mt-8 space-y-6 animate-in fade-in slide-in-from-bottom-4">
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm font-medium text-slate-400">
                  <span className="capitalize flex items-center gap-2">
                    <Activity size={16} className={status === 'processing' ? 'animate-pulse text-orange-400' : ''} />
                    {status}...
                  </span>
                  <span>{progress}%</span>
                </div>
                <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-orange-500 to-yellow-400 transition-all duration-500 ease-out"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="bg-slate-700/50 p-4 rounded-xl text-center border border-slate-600">
                  <p className="text-slate-400 text-xs uppercase tracking-wider">Shots</p>
                  <p className="text-3xl font-bold text-white mt-1">{stats.shots}</p>
                </div>
                <div className="bg-slate-700/50 p-4 rounded-xl text-center border border-slate-600">
                  <p className="text-slate-400 text-xs uppercase tracking-wider">Baskets</p>
                  <p className="text-3xl font-bold text-green-400 mt-1">{stats.baskets}</p>
                </div>
                <div className="bg-slate-700/50 p-4 rounded-xl text-center border border-slate-600">
                  <p className="text-slate-400 text-xs uppercase tracking-wider">Accuracy</p>
                  <p className="text-3xl font-bold text-yellow-400 mt-1">
                    {typeof stats.accuracy === 'number' ? stats.accuracy.toFixed(1) : 0}%
                  </p>
                </div>
              </div>

              {status === 'completed' && (
                <div className="flex justify-center pt-4">
                  <button 
                    onClick={downloadVideo}
                    className="flex items-center gap-2 bg-green-500 hover:bg-green-600 text-white px-8 py-3 rounded-xl font-bold transition-all shadow-lg shadow-green-500/20 transform hover:scale-105"
                  >
                    <Download size={20} /> Download Analyzed Video
                  </button>
                </div>
              )}

              {status === 'error' && (
                <div className="bg-red-500/10 border border-red-500/50 text-red-200 p-4 rounded-lg flex items-center gap-3">
                  <AlertCircle size={20} />
                  <p>{errorMsg || "An unknown error occurred."}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}