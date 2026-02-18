import { useState } from 'react';
import Card from '../components/common/Card';
import Button from '../components/common/Button';
import Modal from '../components/common/Modal';
import { formatDate } from '../utils/helpers';
import { ALERT_LEVEL_COLORS } from '../utils/constants';

const Alerts = () => {
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [filter, setFilter] = useState('all');

  const alerts = [
    {
      id: 1,
      title: 'Unauthorized Vehicle Detected',
      description: 'Unknown vehicle detected in restricted area',
      level: 'critical',
      camera: 'Back Exit',
      timestamp: new Date().toISOString(),
      status: 'active',
      image: null,
    },
    {
      id: 2,
      title: 'Person Loitering',
      description: 'Person detected standing in one location for extended period',
      level: 'high',
      camera: 'Side Gate',
      timestamp: new Date(Date.now() - 600000).toISOString(),
      status: 'active',
      image: null,
    },
    {
      id: 3,
      title: 'Multiple Vehicles',
      description: 'High traffic detected at entrance',
      level: 'medium',
      camera: 'Front Entrance',
      timestamp: new Date(Date.now() - 1200000).toISOString(),
      status: 'resolved',
      image: null,
    },
    {
      id: 4,
      title: 'Camera Obstruction',
      description: 'Camera view partially blocked',
      level: 'high',
      camera: 'Parking Lot',
      timestamp: new Date(Date.now() - 1800000).toISOString(),
      status: 'active',
      image: null,
    },
    {
      id: 5,
      title: 'Motion After Hours',
      description: 'Movement detected outside operating hours',
      level: 'critical',
      camera: 'Front Entrance',
      timestamp: new Date(Date.now() - 2400000).toISOString(),
      status: 'resolved',
      image: null,
    },
    {
      id: 6,
      title: 'Low Confidence Detection',
      description: 'Object detected with low confidence score',
      level: 'low',
      camera: 'Side Gate',
      timestamp: new Date(Date.now() - 3000000).toISOString(),
      status: 'dismissed',
      image: null,
    },
  ];

  const filteredAlerts = alerts.filter((alert) => {
    if (filter === 'all') return true;
    return alert.status === filter;
  });

  const handleViewDetails = (alert) => {
    setSelectedAlert(alert);
    setShowModal(true);
  };

  const getLevelIcon = (level) => {
    if (level === 'critical') {
      return (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      );
    }
    return (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
      </svg>
    );
  };

  const statusCounts = {
    all: alerts.length,
    active: alerts.filter((a) => a.status === 'active').length,
    resolved: alerts.filter((a) => a.status === 'resolved').length,
    dismissed: alerts.filter((a) => a.status === 'dismissed').length,
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Alerts</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Monitor and manage security alerts
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {Object.entries(statusCounts).map(([status, count]) => (
          <button
            key={status}
            onClick={() => setFilter(status)}
            className={`p-4 rounded-lg border-2 transition-all ${
              filter === status
                ? 'border-primary-600 bg-primary-50 dark:bg-primary-900/20'
                : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:border-gray-300 dark:hover:border-gray-600'
            }`}
          >
            <p className="text-sm font-medium text-gray-600 dark:text-gray-400 capitalize">
              {status}
            </p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{count}</p>
          </button>
        ))}
      </div>

      <Card>
        <div className="space-y-4">
          {filteredAlerts.map((alert) => (
            <div
              key={alert.id}
              className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-4 flex-1">
                  <div className={`p-2 rounded-lg ${ALERT_LEVEL_COLORS[alert.level]}`}>
                    {getLevelIcon(alert.level)}
                  </div>

                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        {alert.title}
                      </h3>
                      <span
                        className={`px-2 py-1 rounded-full text-xs font-medium capitalize ${ALERT_LEVEL_COLORS[alert.level]}`}
                      >
                        {alert.level}
                      </span>
                    </div>

                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      {alert.description}
                    </p>

                    <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                      <span className="flex items-center">
                        <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                        {alert.camera}
                      </span>
                      <span className="flex items-center">
                        <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        {formatDate(alert.timestamp)}
                      </span>
                      <span
                        className={`px-2 py-1 rounded-full text-xs font-medium ${
                          alert.status === 'active'
                            ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                            : alert.status === 'resolved'
                            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                            : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
                        }`}
                      >
                        {alert.status}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex space-x-2">
                  <Button size="sm" variant="outline" onClick={() => handleViewDetails(alert)}>
                    View
                  </Button>
                  {alert.status === 'active' && (
                    <>
                      <Button size="sm" variant="success">
                        Resolve
                      </Button>
                      <Button size="sm" variant="ghost">
                        Dismiss
                      </Button>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}

          {filteredAlerts.length === 0 && (
            <div className="text-center py-12">
              <svg
                className="w-16 h-16 text-gray-400 mx-auto mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1}
                  d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
                />
              </svg>
              <p className="text-gray-500 dark:text-gray-400">No alerts found</p>
            </div>
          )}
        </div>
      </Card>

      <Modal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        title="Alert Details"
        size="lg"
      >
        {selectedAlert && (
          <div className="space-y-4">
            <div className="aspect-video bg-gray-900 rounded-lg flex items-center justify-center">
              <div className="text-center text-gray-400">
                <svg className="w-16 h-16 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p className="text-sm">Snapshot unavailable</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Alert Level</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white capitalize">
                  {selectedAlert.level}
                </p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Status</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white capitalize">
                  {selectedAlert.status}
                </p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Camera</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {selectedAlert.camera}
                </p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Time</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {formatDate(selectedAlert.timestamp)}
                </p>
              </div>
            </div>

            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                Description
              </p>
              <p className="text-gray-900 dark:text-white">{selectedAlert.description}</p>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default Alerts;
