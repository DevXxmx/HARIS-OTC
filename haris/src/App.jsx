import React from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import Navbar from './components/Navbar';
import LivePage from './pages/LivePage';
import AlertsPage from './pages/AlertsPage';

function AnimatedRoutes() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<PageTransition><LivePage /></PageTransition>} />
        <Route path="/live" element={<PageTransition><LivePage /></PageTransition>} />

        <Route path="/alerts" element={<PageTransition><AlertsPage /></PageTransition>} />
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
      <div className="min-h-screen flex flex-col font-rajdhani">
        <Navbar />
        <main className="flex-1 w-full relative">
          <AnimatedRoutes />
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
