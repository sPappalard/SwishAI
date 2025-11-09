import { useState, useRef, useEffect } from 'react';
import { Upload, Video, Loader2, Download, Home } from 'lucide-react';

export default function App() {
  const [file, setFile] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [status, setStatus] = useState('idle');
  const [dragActive, setDragActive] = useState(false);
  const [testMode, setTestMode] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0, percentage: 0 });
  const fileInputRef = useRef(null);
  const pollingInterval = useRef(null);
  const [showStopModal, setShowStopModal] = useState(false);

  const API_URL = 'http://localhost:8000';

  useEffect(() => {
    return () => {
      if (pollingInterval.current) clearInterval(pollingInterval.current);
    };
  }, []);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (selectedFile) => {
    if (selectedFile && selectedFile.type.startsWith('video/')) {
      setFile(selectedFile);
      setStatus('idle');
    } else {
      alert('Please select a valid video file');
    }
  };

  const uploadVideo = async () => {
    if (!file) return;
    
    setStatus('uploading');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setFileId(data.file_id);
      setStatus('uploaded');
    } catch (err) {
      alert('Upload failed: ' + err.message);
      setStatus('idle');
    }
  };

  const startPolling = (id) => {
    pollingInterval.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/status/${id}`);
        const data = await res.json();
        
        if (data.status === 'completed') {
          clearInterval(pollingInterval.current);
          setStatus('completed');
          setProgress({ current: data.total, total: data.total, percentage: 100 });
        } else if (data.status === 'stopped') {  // NUOVO
        clearInterval(pollingInterval.current);
        setStatus('stopped');
        setProgress({
          current: data.progress,
          total: data.total,
          percentage: data.percentage || 0
        });
        } else if (data.status === 'processing') {
          setProgress({
            current: data.progress,
            total: data.total,
            percentage: data.percentage || 0
          });
        } else if (data.status === 'error') {
          clearInterval(pollingInterval.current);
          alert('Processing failed: ' + data.message);
          setStatus('uploaded');
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 1000);
  };

  const processVideo = async () => {
    if (!fileId) return;
    
    setStatus('processing');
    setProgress({ current: 0, total: 0, percentage: 0 });
    
    try {
      await fetch(`${API_URL}/process/${fileId}?test_mode=${testMode}`, { method: 'POST' });
      startPolling(fileId);
    } catch (err) {
      alert('Processing failed: ' + err.message);
      setStatus('uploaded');
    }
  };

  const stopProcessing = async () => {
    if (!fileId) return;
    
    try {
      await fetch(`${API_URL}/stop/${fileId}`, { method: 'POST' });
      
      // Ferma il polling
      if (pollingInterval.current) {
        clearInterval(pollingInterval.current);
      }
      
      setStatus('stopped');
      setShowStopModal(false); // Chiudi modale
    } catch (err) {
      alert('Failed to stop processing: ' + err.message);
      setShowStopModal(false);
    }
  };

  const resetApp = () => {
    if (pollingInterval.current) clearInterval(pollingInterval.current);
    if (fileId) {
      fetch(`${API_URL}/clear/${fileId}`, { method: 'DELETE' });
    }
    setFile(null);
    setFileId(null);
    setStatus('idle');
    setTestMode(false);
    setProgress({ current: 0, total: 0, percentage: 0 });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-500 via-orange-600 to-red-600 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-2">🏀 Basketball Tracker</h1>
          <p className="text-orange-100">AI-powered ball, basket and player tracking</p>
        </div>

        <div className="bg-white rounded-2xl shadow-2xl p-8">
          {status === 'idle' || status === 'uploading' ? (
            <div>
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={`border-4 border-dashed rounded-xl p-12 text-center transition-all ${
                  dragActive ? 'border-orange-500 bg-orange-50' : 'border-gray-300 bg-gray-50'
                }`}
              >
                <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                <p className="text-xl font-semibold mb-2">Drop your video here</p>
                <p className="text-gray-500 mb-4">or click to browse (max 3 minutes)</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  onChange={(e) => handleFileSelect(e.target.files[0])}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current.click()}
                  className="bg-orange-500 text-white px-6 py-3 rounded-lg hover:bg-orange-600 transition"
                >
                  Select Video
                </button>
              </div>

              {file && (
                <div className="mt-6 p-4 bg-green-50 rounded-lg flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Video className="w-6 h-6 text-green-600" />
                    <span className="font-medium">{file.name}</span>
                  </div>
                  <button
                    onClick={uploadVideo}
                    disabled={status === 'uploading'}
                    className="bg-orange-500 text-white px-6 py-2 rounded-lg hover:bg-orange-600 disabled:opacity-50 transition"
                  >
                    {status === 'uploading' ? 'Uploading...' : 'Upload & Continue'}
                  </button>
                </div>
              )}
            </div>
          ) : status === 'uploaded' ? (
            <div className="text-center">
              <Video className="w-20 h-20 mx-auto mb-4 text-orange-500" />
              <h2 className="text-2xl font-bold mb-4">Video Ready!</h2>
              <p className="text-gray-600 mb-6">Configure processing options below</p>
              
              <div className="mb-6 flex items-center justify-center gap-3">
                <input
                  type="checkbox"
                  id="testMode"
                  checked={testMode}
                  onChange={(e) => setTestMode(e.target.checked)}
                  className="w-5 h-5 text-orange-500"
                />
                <label htmlFor="testMode" className="text-gray-700 font-medium">
                  Test mode (process only first 15 seconds)
                </label>
              </div>

              <button
                onClick={processVideo}
                className="bg-orange-500 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-orange-600 transition"
              >
                🎯 Process Video
              </button>
            </div>
          ) : status === 'processing' ? (
            <div className="text-center py-12">
              <Loader2 className="w-20 h-20 mx-auto mb-4 text-orange-500 animate-spin" />
              <h2 className="text-2xl font-bold mb-2">Processing Your Video...</h2>
              <p className="text-gray-600 mb-6">
                {testMode ? 'Processing first 15 seconds (test mode)' : 'This may take several minutes'}
              </p>
              
              <div className="max-w-md mx-auto">
                <div className="mb-2 flex justify-between text-sm font-medium">
                  <span>Progress</span>
                  <span>{progress.percentage}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                  <div
                    className="bg-orange-500 h-4 rounded-full transition-all duration-300"
                    style={{ width: `${progress.percentage}%` }}
                  />
                </div>
                <p className="text-sm text-gray-500 mt-2">
                  Frame {progress.current} / {progress.total}
                </p>
              </div>

              {/*Pulsante Stop */}
              <button
                onClick={() => setShowStopModal(true)}
                className="mt-8 bg-red-500 text-white px-8 py-3 rounded-lg hover:bg-red-600 transition font-semibold flex items-center gap-2 mx-auto"
              >
                <span className="text-xl">🛑</span>
                Stop Processing
              </button>
            </div>
          ) : status === 'completed' ? (
            <div className="text-center">
              <div className="mb-6">
                <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-4xl">✅</span>
                </div>
                <h2 className="text-2xl font-bold mb-2">Processing Complete!</h2>
                <p className="text-gray-600">Your tracked video is ready to download</p>
              </div>
              
              <div className="flex gap-4 justify-center">
                <a
                  href={`${API_URL}/download/${fileId}`}
                  download
                  className="bg-green-500 text-white px-8 py-4 rounded-lg hover:bg-green-600 transition flex items-center gap-2 text-lg font-semibold"
                >
                  <Download className="w-6 h-6" />
                  Download Video
                </a>
                <button
                  onClick={resetApp}
                  className="bg-gray-500 text-white px-8 py-4 rounded-lg hover:bg-gray-600 transition flex items-center gap-2 text-lg font-semibold"
                >
                  <Home className="w-6 h-6" />
                  New Video
                </button>
              </div>
            </div>
          ) 
          : status === 'stopped' ? (
            <div className="text-center">
              <div className="mb-6">
                <div className="w-20 h-20 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-4xl">🛑</span>
                </div>
                <h2 className="text-2xl font-bold mb-2">Processing Stopped</h2>
                <p className="text-gray-600">The video processing was interrupted</p>
                <p className="text-sm text-gray-500 mt-2">
                  Processed {progress.current} of {progress.total} frames ({progress.percentage}%)
                </p>
              </div>
              
              <button
                onClick={resetApp}
                className="bg-orange-500 text-white px-8 py-4 rounded-lg hover:bg-orange-600 transition flex items-center gap-2 text-lg font-semibold mx-auto"
              >
                <Home className="w-6 h-6" />
                Start New Video
              </button>
            </div>
          ) : null}
        </div>
      </div>
      {/* NUOVO: Modale conferma stop */}
      {showStopModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-8 transform transition-all">
            <div className="text-center mb-6">
              <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-4xl">⚠️</span>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">
                Stop Processing?
              </h3>
              <p className="text-gray-600">
                Are you sure you want to stop the video processing?
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Progress will be lost and you'll need to start over.
              </p>
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => setShowStopModal(false)}
                className="flex-1 bg-gray-200 text-gray-800 px-6 py-3 rounded-lg hover:bg-gray-300 transition font-semibold"
              >
                Cancel
              </button>
              <button
                onClick={stopProcessing}
                className="flex-1 bg-red-500 text-white px-6 py-3 rounded-lg hover:bg-red-600 transition font-semibold"
              >
                Yes, Stop
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}