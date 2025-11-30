import React, { useState, useEffect, useRef } from 'react';
import { Upload, Play, Square, Download, Activity, CheckCircle, AlertCircle, Video, Settings, Info, RefreshCw, Github, Database, Sun, Clock, X, RotateCcw, Save } from 'lucide-react';

const API_URL = 'http://localhost:8000'; // Change if deployed

// Default thresholds matching the backend
const DEFAULT_THRESHOLDS = {
  0: 0.6,     // Ball
  1: 0.25,    // Ball in Basket
  2: 0.7,     // Player
  3: 0.7,     // Basket
  4: 0.77     // Player Shooting
};

const LABELS = {
  0: "Ball (Orange)",
  1: "Ball in Basket (Gold)",
  2: "Player (Green)",
  3: "Basket (Red)",
  4: "Player Shooting (Blue)"
};

export default function App() {
  // file chosen
  const [file, setFile] = useState(null);
  const [fileId, setFileId] = useState(null);
  // what is happening? (stop, upload, processing)
  const [status, setStatus] = useState('idle');
 // load bar (0-100)
  const [progress, setProgress] = useState(0);
   // scores (shots, baskets)
  const [stats, setStats] = useState({ shots: 0, baskets: 0, accuracy: 0 });
  
  // Settings (test mode, processing mode)
  const [testMode, setTestMode] = useState(false);
  const [processingMode, setProcessingMode] = useState('full_tracking');
  
  // Advanced Settings State
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [thresholds, setThresholds] = useState(DEFAULT_THRESHOLDS);
  
  const [errorMsg, setErrorMsg] = useState('');
  const statusInterval = useRef(null);

  // if the status is "Processing", start a timer that goes off every 1 second: every second it calls the function checkStatus -> when the status changes (es. becomes completed) turns off the timer to avoid consuming resourses 
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
      setErrorMsg('');
    }
  };

  // Function to reset the app state to allow processing a new video (Requested Feature)
  const resetApp = () => {
    setFile(null);
    setFileId(null);
    setStatus('idle');
    setProgress(0);
    setStats({ shots: 0, baskets: 0, accuracy: 0 });
    setErrorMsg('');
    setTestMode(false);
  };
  
  const resetThresholds = () => {
    setThresholds(DEFAULT_THRESHOLDS);
  };
  
  const updateThreshold = (key, value) => {
    setThresholds(prev => ({
      ...prev,
      [key]: parseFloat(value)
    }));
  };

  // there are 2 steps:
  // step 1: UPLOAD --> Sends the raw file to the server. The server responds with an ID (e.g., "video-123")
  // step 2: START PROCESS --> Uses that ID to tell the server, "Okay, now parse video 'video-123' with these settings." As soon as the server responds "Okay, I've started," the Frontend sets the status to processing. Note: This state change automatically triggers the timer (seen above)
  const uploadAndStart = async () => {
    if (!file) return;

    try {
      setStatus('uploading');
      setErrorMsg(''); // Clear previous errors
      const formData = new FormData();
      formData.append('file', file);

      const uploadRes = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
      if (!uploadRes.ok) throw new Error("Upload failed. Check connection or file size.");
      const uploadData = await uploadRes.json();
      const fid = uploadData.file_id;
      setFileId(fid);

      // Pass thresholds as JSON string query param
      const thresholdParam = encodeURIComponent(JSON.stringify(thresholds));

      // Include processing_mode in query
      const processRes = await fetch(
        `${API_URL}/process/${fid}?test_mode=${testMode}&mode=${processingMode}&thresholds=${thresholdParam}`, 
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


  // Call the Python server's /status endpoint.
  // If the server says "I'm at 45%", the Frontend updates the bar (setProgress) and numbers (setStats). The user sees the numbers go up live.
  // If the server says "completed", the Frontend displays the green button to download.
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


  // Graphic Interface (JSX)
  return (
    <div className="min-h-screen bg-slate-900 text-white font-sans flex flex-col justify-between relative">
      
      {/* Advanced Settings Modal */}
      {showAdvancedSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/80 backdrop-blur-sm">
          <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 w-full max-w-lg shadow-2xl animate-in fade-in zoom-in-95">
            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-2 text-orange-400">
                <Settings size={24} />
                <h2 className="text-xl font-bold">Advanced Model Settings</h2>
              </div>
              <button 
                onClick={() => setShowAdvancedSettings(false)}
                className="text-slate-400 hover:text-white transition-colors"
              >
                <X size={24} />
              </button>
            </div>

            <div className="mb-6 bg-yellow-500/10 border border-yellow-500/30 p-4 rounded-lg">
              <p className="text-sm text-yellow-200">
                <span className="font-bold">⚠️ Warning:</span> Changing these values affects AI accuracy. 
                <br/>• <strong>Lower</strong> values detect more objects but may cause false positives.
                <br/>• <strong>Higher</strong> values are stricter but might miss shots or balls.
                <br/>Default values are recommended.
              </p>
            </div>

            <div className="space-y-6">
              {Object.keys(DEFAULT_THRESHOLDS).map((key) => (
                <div key={key} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium text-slate-300">{LABELS[key]}</span>
                    <span className="font-mono text-orange-400">{thresholds[key]}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.95"
                    step="0.01"
                    value={thresholds[key]}
                    onChange={(e) => updateThreshold(key, e.target.value)}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500 hover:accent-orange-400"
                  />
                </div>
              ))}
            </div>

            <div className="flex gap-4 mt-8 pt-6 border-t border-slate-700">
              <button
                onClick={resetThresholds}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
              >
                <RotateCcw size={16} /> Reset Defaults
              </button>
              <div className="flex-1"></div>
              <button
                onClick={() => setShowAdvancedSettings(false)}
                className="flex items-center gap-2 px-6 py-2 bg-orange-500 hover:bg-orange-600 text-white font-bold rounded-lg shadow-lg shadow-orange-500/20 transition-all"
              >
                <Save size={18} /> Save & Close
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="p-8">
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

          {/* Guide / Requirements Card (Requested Feature) */}
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
             <div className="flex items-center gap-2 mb-4 text-orange-400">
               <Info size={20} />
               <h3 className="font-bold text-sm uppercase tracking-wide">For Best Results</h3>
             </div>
             <div className="grid md:grid-cols-3 gap-6">
               <div className="flex items-start gap-3">
                 <div className="bg-slate-700 p-2 rounded-lg text-yellow-400"><Sun size={20} /></div>
                 <div>
                   <p className="font-bold text-sm">Good Lighting</p>
                   <p className="text-xs text-slate-400 mt-1">Ensure the court is well lit and avoid strong backlights.</p>
                 </div>
               </div>
               <div className="flex items-start gap-3">
                 <div className="bg-slate-700 p-2 rounded-lg text-green-400"><Video size={20} /></div>
                 <div>
                   <p className="font-bold text-sm">Clear View</p>
                   <p className="text-xs text-slate-400 mt-1">The camera must clearly see the hoop and the shooter.</p>
                 </div>
               </div>
               <div className="flex items-start gap-3">
                 <div className="bg-slate-700 p-2 rounded-lg text-red-400"><Clock size={20} /></div>
                 <div>
                   <p className="font-bold text-sm">Max Duration</p>
                   <p className="text-xs text-slate-400 mt-1">Videos limited to 3 minutes (180s) for optimal processing.</p>
                 </div>
               </div>
             </div>
          </div>

          {/* Main Card */}
          <div className="bg-slate-800 rounded-2xl p-8 shadow-xl border border-slate-700">
            
            {/* Upload Section */}
            {!file && (
              <div className="flex flex-col items-center justify-center border-2 border-dashed border-slate-600 rounded-xl p-12 transition-all hover:border-orange-500/50 hover:bg-slate-800/50 group">
                <input 
                  type="file" 
                  accept="video/*" 
                  onChange={handleFileChange} 
                  className="hidden" 
                  id="video-upload"
                  disabled={status === 'processing'}
                />
                <label htmlFor="video-upload" className="cursor-pointer flex flex-col items-center gap-4 w-full h-full">
                  <div className="w-20 h-20 bg-slate-700 group-hover:bg-slate-600 rounded-full flex items-center justify-center text-orange-400 transition-colors">
                    <Upload size={40} />
                  </div>
                  <div className="text-center">
                    <p className="text-xl font-bold">Click to Upload Video</p>
                    <p className="text-sm text-slate-500 mt-2">Supports MP4, MOV, AVI</p>
                  </div>
                </label>
              </div>
            )}

            {/* Controls */}
            {file && (
              <div className="mt-2 flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4">
                
                {/* File Info */}
                <div className="flex items-center justify-between bg-slate-900/50 p-4 rounded-xl border border-slate-700">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-orange-500/20 rounded-lg flex items-center justify-center text-orange-400">
                      <Video size={24} />
                    </div>
                    <div>
                       <p className="font-medium text-white truncate max-w-[200px]">{file.name}</p>
                       <p className="text-xs text-slate-400">Ready to process</p>
                    </div>
                  </div>
                  {status !== 'processing' && status !== 'completed' && (
                     <div className="flex gap-4">
                       <button onClick={() => setShowAdvancedSettings(true)} className="flex items-center gap-1 text-xs text-slate-400 hover:text-white transition-colors">
                          <Settings size={14} /> Advanced
                       </button>
                       <button onClick={resetApp} className="text-xs text-slate-400 hover:text-white underline">Change File</button>
                     </div>
                  )}
                </div>
                
                {/* Settings Row (Hidden during processing to keep UI clean) */}
                {status === 'idle' && (
                  <div className="flex flex-wrap items-center justify-center gap-6 bg-slate-900/30 p-4 rounded-xl border border-slate-700/50">
                    
                    {/* Mode Selector */}
                    <div className="flex flex-col gap-2">
                      <label className="text-xs text-slate-400 font-bold uppercase tracking-wider flex items-center gap-2">
                        <Settings size={14} /> Processing Mode
                      </label>
                      <select
                        value={processingMode}
                        onChange={(e) => setProcessingMode(e.target.value)}
                        className="bg-slate-800 text-white border border-slate-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-orange-500 transition-colors"
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
                          className="w-5 h-5 accent-orange-500 rounded focus:ring-orange-500 focus:ring-2"
                        />
                        <span className="text-sm group-hover:text-white transition-colors">Test Mode (15 sec)</span>
                      </label>
                    </div>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex justify-center">
                  {status === 'processing' || status === 'uploading' ? (
                    <div className="flex flex-col items-center gap-4 w-full">
                       {status === 'uploading' && <span className="text-orange-400 animate-pulse font-bold">Uploading Video... Please Wait</span>}
                       {status === 'processing' && (
                        <button 
                          onClick={stopProcessing}
                          className="flex items-center gap-2 bg-red-500 hover:bg-red-600 text-white px-8 py-3 rounded-xl font-bold transition-all shadow-lg shadow-red-500/20"
                        >
                          <Square size={20} /> Stop Analysis
                        </button>
                       )}
                    </div>
                  ) : status === 'completed' ? (
                    <div className="flex flex-wrap gap-4 justify-center w-full">
                        <button 
                          onClick={downloadVideo}
                          className="flex items-center gap-2 bg-green-500 hover:bg-green-600 text-white px-8 py-3 rounded-xl font-bold transition-all shadow-lg shadow-green-500/20 transform hover:scale-105"
                        >
                          <Download size={20} /> Download Result
                        </button>
                        {/* New Feature: Process another video */}
                        <button 
                          onClick={resetApp}
                          className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-white px-8 py-3 rounded-xl font-bold transition-all shadow-lg"
                        >
                          <RefreshCw size={20} /> Process Another Video
                        </button>
                    </div>
                  ) : (
                    <button 
                      onClick={uploadAndStart}
                      disabled={status === 'error'}
                      className="flex items-center gap-2 bg-orange-500 hover:bg-orange-600 text-white px-12 py-4 text-lg rounded-xl font-bold transition-all shadow-lg shadow-orange-500/20 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105"
                    >
                      <Play size={24} /> Start Analysis
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
                      <Activity size={16} className={status === 'processing' || status === 'uploading' ? 'animate-pulse text-orange-400' : ''} />
                      {status === 'uploading' ? 'Uploading...' : status === 'processing' ? 'Analyzing...' : status}
                    </span>
                    <span>{progress}%</span>
                  </div>
                  <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-500 ease-out ${status === 'error' ? 'bg-red-500' : 'bg-gradient-to-r from-orange-500 to-yellow-400'}`}
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

                {/* Friendly Error Display */}
                {status === 'error' && (
                  <div className="bg-red-500/10 border border-red-500/50 text-red-200 p-6 rounded-xl flex items-center gap-4 animate-in shake">
                    <AlertCircle size={32} className="text-red-500 flex-shrink-0" />
                    <div>
                      <h4 className="font-bold text-red-400 text-lg">Oops! Something went wrong.</h4>
                      <p>{errorMsg || "An unknown error occurred. Please try a different video."}</p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer / Credits (Requested Feature) */}
      <footer className="w-full p-6 text-center text-slate-500 text-sm border-t border-slate-800 mt-12 bg-slate-900/80 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="flex items-center gap-2">
            Built with <span className="text-red-500">♥</span> by 
            <a href="https://github.com/sPappalard" target="_blank" rel="noopener noreferrer" className="text-slate-300 hover:text-orange-400 flex items-center gap-1 transition-colors font-medium">
              <Github size={14} /> sPappalard
            </a>
          </p>
          <div className="flex items-center gap-4">
            <a href="https://universe.roboflow.com/basketball-6vyfz/basketball-detection-srfkd" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 hover:text-white transition-colors">
              <Database size={14} /> Roboflow Dataset
            </a>
            <span className="text-slate-700">|</span>
            <span>v1.0.1</span>
          </div>
        </div>
      </footer>
    </div>
  );
}