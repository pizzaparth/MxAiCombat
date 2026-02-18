import { useState, useMemo } from 'react';
import Card from '../components/common/Card';
import Input from '../components/common/Input';
import Button from '../components/common/Button';
import { formatDate } from '../utils/helpers';

const Logs = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [filterType, setFilterType] = useState('all');
  const itemsPerPage = 10;

  const allLogs = [
    { id: 1, timestamp: new Date().toISOString(), camera: 'Front Entrance', type: 'Vehicle', confidence: 95, details: 'Sedan, Blue', status: 'verified' },
    { id: 2, timestamp: new Date(Date.now() - 300000).toISOString(), camera: 'Parking Lot', type: 'Person', confidence: 88, details: 'Walking', status: 'pending' },
    { id: 3, timestamp: new Date(Date.now() - 600000).toISOString(), camera: 'Back Exit', type: 'Vehicle', confidence: 92, details: 'SUV, Black', status: 'verified' },
    { id: 4, timestamp: new Date(Date.now() - 900000).toISOString(), camera: 'Side Gate', type: 'Person', confidence: 85, details: 'Standing', status: 'verified' },
    { id: 5, timestamp: new Date(Date.now() - 1200000).toISOString(), camera: 'Front Entrance', type: 'Vehicle', confidence: 97, details: 'Truck, White', status: 'verified' },
    { id: 6, timestamp: new Date(Date.now() - 1500000).toISOString(), camera: 'Parking Lot', type: 'Animal', confidence: 78, details: 'Cat', status: 'false_positive' },
    { id: 7, timestamp: new Date(Date.now() - 1800000).toISOString(), camera: 'Back Exit', type: 'Person', confidence: 90, details: 'Running', status: 'verified' },
    { id: 8, timestamp: new Date(Date.now() - 2100000).toISOString(), camera: 'Side Gate', type: 'Vehicle', confidence: 94, details: 'Van, Gray', status: 'verified' },
    { id: 9, timestamp: new Date(Date.now() - 2400000).toISOString(), camera: 'Front Entrance', type: 'Person', confidence: 82, details: 'Walking', status: 'pending' },
    { id: 10, timestamp: new Date(Date.now() - 2700000).toISOString(), camera: 'Parking Lot', type: 'Vehicle', confidence: 96, details: 'Sedan, Red', status: 'verified' },
    { id: 11, timestamp: new Date(Date.now() - 3000000).toISOString(), camera: 'Back Exit', type: 'Unknown', confidence: 65, details: 'Unclear', status: 'review' },
    { id: 12, timestamp: new Date(Date.now() - 3300000).toISOString(), camera: 'Side Gate', type: 'Person', confidence: 91, details: 'Walking', status: 'verified' },
    { id: 13, timestamp: new Date(Date.now() - 3600000).toISOString(), camera: 'Front Entrance', type: 'Vehicle', confidence: 93, details: 'Motorcycle', status: 'verified' },
    { id: 14, timestamp: new Date(Date.now() - 3900000).toISOString(), camera: 'Parking Lot', type: 'Person', confidence: 87, details: 'Standing', status: 'verified' },
    { id: 15, timestamp: new Date(Date.now() - 4200000).toISOString(), camera: 'Back Exit', type: 'Vehicle', confidence: 89, details: 'Sedan, Silver', status: 'verified' },
  ];

  const filteredLogs = useMemo(() => {
    return allLogs.filter((log) => {
      const matchesSearch =
        log.camera.toLowerCase().includes(searchTerm.toLowerCase()) ||
        log.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
        log.details.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesFilter = filterType === 'all' || log.type.toLowerCase() === filterType.toLowerCase();

      return matchesSearch && matchesFilter;
    });
  }, [searchTerm, filterType]);

  const totalPages = Math.ceil(filteredLogs.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentLogs = filteredLogs.slice(startIndex, endIndex);

  const getStatusBadge = (status) => {
    const styles = {
      verified: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
      pending: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
      false_positive: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
      review: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    };

    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${styles[status]}`}>
        {status.replace('_', ' ')}
      </span>
    );
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Detection Logs</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Search and analyze detection history
        </p>
      </div>

      <Card>
        <div className="space-y-4">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <Input
                placeholder="Search by camera, type, or details..."
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setCurrentPage(1);
                }}
                icon={
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                }
              />
            </div>

            <div className="flex gap-2">
              {['all', 'vehicle', 'person', 'animal'].map((type) => (
                <Button
                  key={type}
                  size="sm"
                  variant={filterType === type ? 'primary' : 'ghost'}
                  onClick={() => {
                    setFilterType(type);
                    setCurrentPage(1);
                  }}
                  className="capitalize"
                >
                  {type}
                </Button>
              ))}
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900 dark:text-white">
                    Timestamp
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900 dark:text-white">
                    Camera
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900 dark:text-white">
                    Type
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900 dark:text-white">
                    Confidence
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900 dark:text-white">
                    Details
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900 dark:text-white">
                    Status
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-900 dark:text-white">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {currentLogs.map((log) => (
                  <tr
                    key={log.id}
                    className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900 transition-colors"
                  >
                    <td className="py-3 px-4 text-sm text-gray-900 dark:text-white font-mono">
                      {formatDate(log.timestamp)}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-900 dark:text-white">
                      {log.camera}
                    </td>
                    <td className="py-3 px-4">
                      <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200">
                        {log.type}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-2">
                        <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2 max-w-[60px]">
                          <div
                            className={`h-2 rounded-full ${
                              log.confidence >= 90
                                ? 'bg-green-600'
                                : log.confidence >= 80
                                ? 'bg-yellow-600'
                                : 'bg-red-600'
                            }`}
                            style={{ width: `${log.confidence}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {log.confidence}%
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600 dark:text-gray-400">
                      {log.details}
                    </td>
                    <td className="py-3 px-4">{getStatusBadge(log.status)}</td>
                    <td className="py-3 px-4">
                      <button className="text-primary-600 hover:text-primary-700 text-sm font-medium">
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {currentLogs.length === 0 && (
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
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                <p className="text-gray-500 dark:text-gray-400">No logs found</p>
              </div>
            )}
          </div>

          {totalPages > 1 && (
            <div className="flex items-center justify-between pt-4">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Showing {startIndex + 1} to {Math.min(endIndex, filteredLogs.length)} of{' '}
                {filteredLogs.length} results
              </p>

              <div className="flex items-center space-x-2">
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={currentPage === 1}
                  onClick={() => setCurrentPage((prev) => prev - 1)}
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                  Previous
                </Button>

                <div className="flex space-x-1">
                  {[...Array(totalPages)].map((_, index) => (
                    <Button
                      key={index}
                      size="sm"
                      variant={currentPage === index + 1 ? 'primary' : 'ghost'}
                      onClick={() => setCurrentPage(index + 1)}
                    >
                      {index + 1}
                    </Button>
                  ))}
                </div>

                <Button
                  size="sm"
                  variant="ghost"
                  disabled={currentPage === totalPages}
                  onClick={() => setCurrentPage((prev) => prev + 1)}
                >
                  Next
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </Button>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default Logs;
