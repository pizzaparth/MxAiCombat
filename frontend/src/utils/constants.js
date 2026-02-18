export const API_BASE_URL = 'http://localhost:3000/api';

export const ROUTES = {
  LOGIN: '/login',
  REGISTER: '/register',
  DASHBOARD: '/',
  CAMERAS: '/cameras',
  LOGS: '/logs',
  ALERTS: '/alerts',
  SETTINGS: '/settings',
};

export const DETECTION_TYPES = {
  VEHICLE: 'vehicle',
  PERSON: 'person',
  ANIMAL: 'animal',
  UNKNOWN: 'unknown',
};

export const ALERT_LEVELS = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical',
};

export const ALERT_LEVEL_COLORS = {
  low: 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900',
  medium: 'text-yellow-600 bg-yellow-100 dark:text-yellow-400 dark:bg-yellow-900',
  high: 'text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900',
  critical: 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900',
};
