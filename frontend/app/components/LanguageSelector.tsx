import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Modal, FlatList } from 'react-native';
import { Languages, ChevronDown } from 'lucide-react-native';
import { useI18n } from '../../lib/i18n';

interface Language {
  code: string;
  name: string;
  nativeName: string;
}

const languages: Language[] = [
  { code: 'en', name: 'English', nativeName: 'English' },
  { code: 'hi', name: 'Hindi', nativeName: 'हिन्दी' },
  { code: 'es', name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ' },
  { code: 'fr', name: 'Gujarati', nativeName: 'ગુજરાતી' },
  { code: 'ar', name: 'Tamil', nativeName: 'தமிழ்' },
  { code: 'pt', name: 'Malayalam', nativeName: 'മലയാളം' },
  { code: 'bn', name: 'Bengali', nativeName: 'বাংলা' },
];

interface LanguageSelectorProps {
  selectedLanguage: string;
  onLanguageChange: (language: string) => void;
}

const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  selectedLanguage,
  onLanguageChange,
}) => {
  const { translations } = useI18n();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const selectedLang = languages.find(lang => lang.code === selectedLanguage);

  const handleLanguageSelect = (language: Language) => {
    onLanguageChange(language.code);
    setIsModalVisible(false);
  };

  const renderLanguageItem = ({ item }: { item: Language }) => (
    <TouchableOpacity
      style={[
        styles.languageItem,
        item.code === selectedLanguage && styles.selectedLanguageItem
      ]}
      onPress={() => handleLanguageSelect(item)}
    >
      <View style={styles.languageItemContent}>
        <Text style={[
          styles.nativeName,
          item.code === selectedLanguage && styles.selectedText
        ]}>
          {item.nativeName}
        </Text>
        <Text style={[
          styles.englishName,
          item.code === selectedLanguage && styles.selectedSubText
        ]}>
          ({item.name})
        </Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Languages size={20} color="#4CAF50" />
        <Text style={styles.headerText}>{translations.selectLanguage}</Text>
      </View>
      
      <TouchableOpacity
        style={styles.selector}
        onPress={() => setIsModalVisible(true)}
      >
        <View style={styles.selectorContent}>
          {selectedLang ? (
            <View>
              <Text style={styles.selectedNativeName}>{selectedLang.nativeName}</Text>
              <Text style={styles.selectedEnglishName}>({selectedLang.name})</Text>
            </View>
          ) : (
            <Text style={styles.placeholder}>{translations.selectLanguage}</Text>
          )}
        </View>
        <ChevronDown size={20} color="#666" />
      </TouchableOpacity>

      <Modal
        visible={isModalVisible}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setIsModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>{translations.chooseLanguage}</Text>
              <TouchableOpacity
                style={styles.closeButton}
                onPress={() => setIsModalVisible(false)}
              >
                <Text style={styles.closeButtonText}>×</Text>
              </TouchableOpacity>
            </View>
            
            <FlatList
              data={languages}
              renderItem={renderLanguageItem}
              keyExtractor={(item) => item.code}
              style={styles.languageList}
            />
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    margin: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  headerText: {
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  selector: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
  },
  selectorContent: {
    flex: 1,
  },
  selectedNativeName: {
    fontSize: 16,
    fontWeight: '600',
  },
  selectedEnglishName: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
  placeholder: {
    fontSize: 16,
    color: '#999',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 12,
    width: '90%',
    maxHeight: '70%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  closeButton: {
    padding: 4,
  },
  closeButtonText: {
    fontSize: 24,
    color: '#666',
  },
  languageList: {
    maxHeight: 400,
  },
  languageItem: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  selectedLanguageItem: {
    backgroundColor: '#f8f9fa',
  },
  languageItemContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  nativeName: {
    fontSize: 16,
    fontWeight: '500',
  },
  englishName: {
    fontSize: 14,
    color: '#666',
  },
  selectedText: {
    color: '#4CAF50',
    fontWeight: '600',
  },
  selectedSubText: {
    color: '#4CAF50',
  },
});

export default LanguageSelector;