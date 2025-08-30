import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { AlertTriangle, CheckCircle, Info, Volume2 } from 'lucide-react-native';

interface DiagnosisData {
  disease?: string;
  confidence?: number;
  severity?: 'low' | 'medium' | 'high';
  recommendations?: string[];
  description?: string;
}

interface DiagnosisResultsProps {
  results: DiagnosisData | null;
  isLoading?: boolean;
  onPlayAudio?: () => void;
}

const DiagnosisResults: React.FC<DiagnosisResultsProps> = ({ 
  results, 
  isLoading, 
  onPlayAudio 
}) => {
  const getSeverityIcon = (severity?: string) => {
    switch (severity) {
      case 'high':
        return <AlertTriangle size={20} color="#f44336" />;
      case 'medium':
        return <AlertTriangle size={20} color="#ff9800" />;
      case 'low':
        return <CheckCircle size={20} color="#4CAF50" />;
      default:
        return <Info size={20} color="#2196F3" />;
    }
  };

  const getSeverityColor = (severity?: string) => {
    switch (severity) {
      case 'high':
        return '#f44336';
      case 'medium':
        return '#ff9800';
      case 'low':
        return '#4CAF50';
      default:
        return '#666';
    }
  };

  if (isLoading) {
    return (
      <View style={styles.container}>
        <View style={styles.loadingContainer}>
          <View style={styles.loadingBar1} />
          <View style={styles.loadingBar2} />
          <View style={styles.loadingBar3} />
          <Text style={styles.loadingText}>Analyzing...</Text>
        </View>
      </View>
    );
  }

  if (!results) {
    return (
      <View style={[styles.container, styles.emptyContainer]}>
        <Info size={48} color="#ccc" style={styles.emptyIcon} />
        <Text style={styles.emptyText}>
          Upload an image or record a voice query to get started
        </Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.titleRow}>
          {getSeverityIcon(results.severity)}
          <Text style={styles.title}>
            {results.disease || 'Diagnosis Complete'}
          </Text>
        </View>
        
        {results.confidence && (
          <View style={styles.confidenceRow}>
            <Text style={styles.confidenceLabel}>Confidence: </Text>
            <View style={styles.confidenceBadge}>
              <Text style={styles.confidenceText}>
                {Math.round(results.confidence * 100)}%
              </Text>
            </View>
          </View>
        )}
        
        {results.severity && (
          <View style={[styles.severityBadge, { backgroundColor: getSeverityColor(results.severity) }]}>
            <Text style={styles.severityText}>
              {results.severity.charAt(0).toUpperCase() + results.severity.slice(1)} Risk
            </Text>
          </View>
        )}
      </View>

      {/* Description */}
      {results.description && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Description</Text>
          <Text style={styles.description}>
            {results.description}
          </Text>
        </View>
      )}

      {/* Recommendations */}
      {results.recommendations && results.recommendations.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Recommendations</Text>
          {results.recommendations.map((recommendation, index) => (
            <View key={index} style={styles.recommendationItem}>
              <CheckCircle size={16} color="#4CAF50" style={styles.checkIcon} />
              <Text style={styles.recommendationText}>{recommendation}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Audio playback */}
      {onPlayAudio && (
        <View style={styles.audioSection}>
          <TouchableOpacity onPress={onPlayAudio} style={styles.audioButton}>
            <Volume2 size={20} color="#2196F3" />
            <Text style={styles.audioButtonText}>Listen to Audio Response</Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    margin: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  loadingContainer: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  loadingBar1: {
    width: '75%',
    height: 20,
    backgroundColor: '#f0f0f0',
    borderRadius: 10,
    marginBottom: 10,
  },
  loadingBar2: {
    width: '50%',
    height: 16,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    marginBottom: 10,
  },
  loadingBar3: {
    width: '60%',
    height: 16,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    marginBottom: 20,
  },
  loadingText: {
    fontSize: 16,
    color: '#666',
    fontStyle: 'italic',
  },
  emptyContainer: {
    alignItems: 'center',
    paddingVertical: 40,
    borderWidth: 2,
    borderStyle: 'dashed',
    borderColor: '#ddd',
  },
  emptyIcon: {
    marginBottom: 16,
  },
  emptyText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  header: {
    marginBottom: 24,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginLeft: 8,
    flex: 1,
  },
  confidenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  confidenceLabel: {
    fontSize: 14,
    color: '#666',
  },
  confidenceBadge: {
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  confidenceText: {
    fontSize: 14,
    fontWeight: '600',
  },
  severityBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  severityText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  description: {
    fontSize: 14,
    color: '#666',
    lineHeight: 22,
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  checkIcon: {
    marginTop: 2,
    marginRight: 8,
  },
  recommendationText: {
    flex: 1,
    fontSize: 14,
    lineHeight: 20,
  },
  audioSection: {
    borderTopWidth: 1,
    borderTopColor: '#eee',
    paddingTop: 20,
  },
  audioButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f8f9fa',
    borderWidth: 1,
    borderColor: '#dee2e6',
    borderRadius: 8,
    padding: 16,
  },
  audioButtonText: {
    marginLeft: 8,
    fontSize: 16,
    color: '#2196F3',
    fontWeight: '600',
  },
});

export default DiagnosisResults;