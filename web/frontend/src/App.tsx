import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { Zap, BarChart2, Terminal, FlaskConical } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Monitor from './pages/Monitor'
import Results from './pages/Results'
import Baselines from './pages/Baselines'

const NAV = [
  { to: '/', icon: Zap, label: 'Запуск' },
  { to: '/monitor', icon: Terminal, label: 'Монитор' },
  { to: '/results', icon: BarChart2, label: 'Результаты' },
  { to: '/baselines', icon: FlaskConical, label: 'Бейзлайны' },
]

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen">
        {/* Sidebar */}
        <nav className="w-48 bg-dark-1 border-r border-gray-800 flex flex-col py-6 gap-1 shrink-0">
          <div className="px-4 mb-6">
            <span className="text-brand font-bold text-xl glow">⚡ EvoHash</span>
          </div>
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-2 mx-2 rounded transition-all text-sm ${
                  isActive
                    ? 'bg-brand/10 text-brand border border-brand/30'
                    : 'text-gray-400 hover:text-white hover:bg-dark-2'
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Main content */}
        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/monitor" element={<Monitor />} />
            <Route path="/results" element={<Results />} />
            <Route path="/baselines" element={<Baselines />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
