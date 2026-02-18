export const mockApi = {
  auth: {
    login: async (email, password) => {
      await new Promise((resolve) => setTimeout(resolve, 800));

      if (email && password) {
        return {
          success: true,
          data: {
            user: {
              id: 1,
              email: email,
              name: 'John Doe',
              avatar: null,
            },
            token: 'mock-jwt-token-' + Date.now(),
          },
        };
      }

      return {
        success: false,
        error: 'Invalid credentials',
      };
    },

    register: async (name, email, password) => {
      await new Promise((resolve) => setTimeout(resolve, 800));

      if (name && email && password) {
        return {
          success: true,
          data: {
            user: {
              id: 1,
              email: email,
              name: name,
              avatar: null,
            },
            token: 'mock-jwt-token-' + Date.now(),
          },
        };
      }

      return {
        success: false,
        error: 'Registration failed',
      };
    },
  },

  dashboard: {
    getStats: async () => {
      await new Promise((resolve) => setTimeout(resolve, 500));

      return {
        success: true,
        data: {
          totalDetections: 1247,
          activeAlerts: 23,
          camerasOnline: 4,
          accuracyRate: 98.5,
        },
      };
    },

    getChartData: async () => {
      await new Promise((resolve) => setTimeout(resolve, 500));

      return {
        success: true,
        data: [
          { name: 'Mon', detections: 45, alerts: 12 },
          { name: 'Tue', detections: 52, alerts: 15 },
          { name: 'Wed', detections: 48, alerts: 10 },
          { name: 'Thu', detections: 61, alerts: 18 },
          { name: 'Fri', detections: 55, alerts: 14 },
          { name: 'Sat', detections: 38, alerts: 8 },
          { name: 'Sun', detections: 42, alerts: 11 },
        ],
      };
    },
  },

  cameras: {
    getAll: async () => {
      await new Promise((resolve) => setTimeout(resolve, 500));

      return {
        success: true,
        data: [
          { id: 1, name: 'Front Entrance', status: 'online', location: 'Building A' },
          { id: 2, name: 'Parking Lot', status: 'online', location: 'Level 1' },
          { id: 3, name: 'Back Exit', status: 'offline', location: 'Building B' },
          { id: 4, name: 'Side Gate', status: 'online', location: 'Perimeter' },
        ],
      };
    },
  },

  logs: {
    getAll: async (page = 1, limit = 10) => {
      await new Promise((resolve) => setTimeout(resolve, 500));

      const allLogs = Array.from({ length: 50 }, (_, i) => ({
        id: i + 1,
        timestamp: new Date(Date.now() - i * 300000).toISOString(),
        camera: ['Front Entrance', 'Parking Lot', 'Back Exit', 'Side Gate'][i % 4],
        type: ['Vehicle', 'Person', 'Animal', 'Unknown'][Math.floor(Math.random() * 4)],
        confidence: 65 + Math.floor(Math.random() * 35),
        details: ['Sedan, Blue', 'Walking', 'Cat', 'SUV, Black'][Math.floor(Math.random() * 4)],
        status: ['verified', 'pending', 'false_positive', 'review'][
          Math.floor(Math.random() * 4)
        ],
      }));

      const startIndex = (page - 1) * limit;
      const endIndex = startIndex + limit;

      return {
        success: true,
        data: {
          logs: allLogs.slice(startIndex, endIndex),
          total: allLogs.length,
          page,
          limit,
        },
      };
    },
  },

  alerts: {
    getAll: async () => {
      await new Promise((resolve) => setTimeout(resolve, 500));

      return {
        success: true,
        data: [
          {
            id: 1,
            title: 'Unauthorized Vehicle Detected',
            description: 'Unknown vehicle detected in restricted area',
            level: 'critical',
            camera: 'Back Exit',
            timestamp: new Date().toISOString(),
            status: 'active',
          },
          {
            id: 2,
            title: 'Person Loitering',
            description: 'Person detected standing in one location for extended period',
            level: 'high',
            camera: 'Side Gate',
            timestamp: new Date(Date.now() - 600000).toISOString(),
            status: 'active',
          },
          {
            id: 3,
            title: 'Multiple Vehicles',
            description: 'High traffic detected at entrance',
            level: 'medium',
            camera: 'Front Entrance',
            timestamp: new Date(Date.now() - 1200000).toISOString(),
            status: 'resolved',
          },
        ],
      };
    },
  },
};

export default mockApi;
