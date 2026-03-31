import React from 'react';
import { BrowserRouter, Routes, Route, useLocation, Navigate } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import Navbar from './components/Navbar';
import LivePage from './pages/LivePage';
import AlertsPage from './pages/AlertsPage';
import AboutPage from './pages/AboutPage';
import LoginPage from './pages/LoginPage';

function PrivateRoute({ children }) {
  const isAuthenticated = localStorage.getItem('isAuthenticated') === 'true';
  return isAuthenticated ? children : <Navigate to="/login" replace />;
}

function AnimatedRoutes() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<PrivateRoute><PageTransition><LivePage /></PageTransition></PrivateRoute>} />
        <Route path="/live" element={<PrivateRoute><PageTransition><LivePage /></PageTransition></PrivateRoute>} />
        <Route path="/alerts" element={<PrivateRoute><PageTransition><AlertsPage /></PageTransition></PrivateRoute>} />
        <Route path="/about" element={<PrivateRoute><PageTransition><AboutPage /></PageTransition></PrivateRoute>} />
        <Route path="/login" element={<PageTransition><LoginPage /></PageTransition>} />
      </Routes>
    </AnimatePresence>
  );
}

function PageTransition({ children }) {
  return (
    <>
      <motion.div
        initial={{ x: "-100%" }}
        animate={{ x: ["-100%", "0%", "100%"] }}
        transition={{ duration: 0.8, ease: "easeInOut", times: [0, 0.5, 1] }}
        className="fixed inset-0 z-50 bg-red-dim pointer-events-none"
      />
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0, transition: { duration: 0.3 } }}
        transition={{ duration: 0.3, delay: 0.4 }}
        className="w-full h-full"
      >
        {children}
      </motion.div>
    </>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

function AppContent() {
  const location = useLocation();
  const showNavbar = location.pathname !== '/login';

  return (
    <div className="min-h-screen flex flex-col font-rajdhani">
      {showNavbar && <Navbar />}
      <main className="flex-1 w-full relative">
        <AnimatedRoutes />
      </main>
    </div>
  );
}

export default App;
