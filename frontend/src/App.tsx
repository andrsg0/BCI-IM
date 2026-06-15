import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AppLayout } from './components/layout/AppLayout'
import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import SignalLab from './pages/SignalLab'
import SpatialCSP from './pages/SpatialCSP'
import Brain3DPage from './pages/Brain3DPage'
import LiveStream from './pages/LiveStream'
import Results from './pages/Results'
import Glossary from './pages/Glossary'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/lab" element={<SignalLab />} />
          <Route path="/csp" element={<SpatialCSP />} />
          <Route path="/brain" element={<Brain3DPage />} />
          <Route path="/live" element={<LiveStream />} />
          <Route path="/results" element={<Results />} />
          <Route path="/glossary" element={<Glossary />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
