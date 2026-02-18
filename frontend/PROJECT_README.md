# AI Car Monitoring System - Frontend Dashboard

A complete production-ready React dashboard for AI-powered car monitoring and surveillance.

## Features

- **Authentication System**: Login/Register pages with protected routes
- **Dashboard**: Real-time statistics with interactive charts (Recharts)
- **Live Camera Monitoring**: Camera feed UI with detection visualization
- **Detection Logs**: Searchable and paginated table of all detections
- **Alerts Management**: Real-time alert monitoring with filtering
- **Settings**: Profile management and theme customization
- **Dark/Light Mode**: Fully implemented theme toggle
- **Responsive Design**: Mobile-first design that works on all screen sizes
- **Toast Notifications**: Real-time feedback system
- **Loading States**: Elegant loading spinners throughout
- **Error Handling**: Comprehensive error handling UI

## Tech Stack

- **React 18** - UI library
- **React Router v6** - Client-side routing
- **Context API** - Global state management
- **Tailwind CSS** - Utility-first styling
- **Axios** - HTTP client with interceptors
- **Recharts** - Chart library for data visualization
- **Vite** - Build tool and dev server

## Project Structure

```
src/
├── components/
│   ├── common/          # Reusable components
│   │   ├── Button.jsx
│   │   ├── Input.jsx
│   │   ├── Modal.jsx
│   │   ├── Card.jsx
│   │   ├── LoadingSpinner.jsx
│   │   └── Toast.jsx
│   ├── layout/          # Layout components
│   │   ├── MainLayout.jsx
│   │   ├── Sidebar.jsx
│   │   └── Navbar.jsx
│   └── ProtectedRoute.jsx
├── contexts/            # React Context providers
│   ├── AuthContext.jsx
│   ├── ThemeContext.jsx
│   └── ToastContext.jsx
├── pages/              # Page components
│   ├── Login.jsx
│   ├── Register.jsx
│   ├── Dashboard.jsx
│   ├── Cameras.jsx
│   ├── Logs.jsx
│   ├── Alerts.jsx
│   └── Settings.jsx
├── services/           # API services
│   ├── axios.js
│   └── mockApi.js
├── utils/             # Utility functions
│   ├── constants.js
│   └── helpers.js
├── App.jsx           # Main app component
└── main.jsx          # Entry point
```

## Installation

```bash
npm install
```

## Running the Application

### Development Mode
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### Production Build
```bash
npm run build
```

### Preview Production Build
```bash
npm run preview
```

## Usage

### Authentication
- Any email/password combination will work (mock authentication)
- The system automatically redirects to dashboard after login
- Protected routes require authentication

### Navigation
- Use the sidebar to navigate between pages
- Click the hamburger menu on mobile to toggle sidebar
- Theme toggle in the navbar switches between light/dark mode

### Mock Data
The application uses mock API services for demo purposes. All data is generated locally and no backend is required.

## Key Components

### Context Providers

**AuthContext**
- Handles authentication state
- Provides login/register/logout functions
- Manages user session

**ThemeContext**
- Manages light/dark theme
- Persists theme preference to localStorage
- Applies theme to root element

**ToastContext**
- Global notification system
- Supports success, error, warning, and info messages
- Auto-dismiss with configurable duration

### Reusable Components

**Button** - Versatile button with multiple variants (primary, secondary, danger, success, outline, ghost)

**Input** - Form input with label, error display, and icon support

**Modal** - Accessible modal dialog with backdrop and keyboard controls

**Card** - Container component with optional title and action buttons

**LoadingSpinner** - Loading indicator with multiple sizes

**Toast** - Notification component with auto-dismiss

### Pages

**Dashboard** - Overview with stats cards and charts
**Cameras** - Live camera monitoring interface
**Logs** - Searchable and paginated detection logs
**Alerts** - Alert management with filtering
**Settings** - User profile and preferences

## Customization

### Colors
Edit `tailwind.config.js` to customize the color scheme:
```javascript
theme: {
  extend: {
    colors: {
      primary: { ... }
    }
  }
}
```

### API Integration
Replace mock API calls in `src/services/mockApi.js` with real API endpoints:
```javascript
import axios from './axios';

export const api = {
  dashboard: {
    getStats: async () => {
      const response = await axios.get('/dashboard/stats');
      return response.data;
    }
  }
};
```

## Environment Variables
Create a `.env` file for environment-specific configuration:
```
VITE_API_URL=http://localhost:3000/api
```

## Browser Support
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License
MIT
