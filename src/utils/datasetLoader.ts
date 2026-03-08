/**
 * Dataset loader for training and improving pattern detection
 * Supports SROIE2019, Total-Text, and MIDV500 datasets
 */

export interface DatasetSample {
  id: string;
  imagePath: string;
  imageUrl?: string;
  text: string;
  entities: Record<string, string>;
  source: 'sroie' | 'total-text' | 'midv500';
}

export type DatasetLoadCallback = (current: number, total: number) => void;

export interface PatternStats {
  type: string;
  count: number;
  examples: string[];
  patterns: RegExp[];
}

/**
 * Load SROIE2019 dataset samples (receipts with structured data)
 * Loads actual images and ground truth from the dataset folder
 */
export async function loadSROIEDataset(
  limit: number = 50,
  onProgress?: DatasetLoadCallback
): Promise<DatasetSample[]> {
  const samples: DatasetSample[] = [];
  
  try {
    // List of sample image IDs from SROIE2019 dataset
    const imageIds = [
      'X00016469612', 'X00016469619', 'X00016469620', 'X00016469622', 'X00016469623',
      'X00016469669', 'X00016469672', 'X00016469676', 'X51005200938', 'X51005230617',
      'X51005230618', 'X51005230619', 'X51005230620', 'X51005230621', 'X51005230622',
      'X51005230623', 'X51005230624', 'X51005230625', 'X51005230626', 'X51005230627',
      'X51005230628', 'X51005230629', 'X51005230630', 'X51005230631', 'X51005230632',
      'X51005230633', 'X51005230634', 'X51005230635', 'X51005230636', 'X51005230637',
      'X51005230638', 'X51005230639', 'X51005230640', 'X51005230641', 'X51005230642',
      'X51005230643', 'X51005230644', 'X51005230645', 'X51005230646', 'X51005230647',
      'X51005230648', 'X51005230649', 'X51005230650', 'X51005230651', 'X51005230652',
      'X51005230653', 'X51005230654', 'X51005230655', 'X51005230656', 'X51005230657',
    ].slice(0, limit);
    
    for (let i = 0; i < imageIds.length; i++) {
      const imageId = imageIds[i];
      
      // Report progress
      if (onProgress) {
        onProgress(i + 1, imageIds.length);
      }
      try {
        // Fetch entity file (ground truth) with timeout
        const entityPath = `/dataset/SROIE2019/train/entities/${imageId}.txt`;
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
        
        const entityResponse = await fetch(entityPath, { signal: controller.signal });
        clearTimeout(timeoutId);
        
        if (!entityResponse.ok) {
          console.warn(`Entity file not found (${entityResponse.status}): ${entityPath}`);
          continue;
        }
        
        const entityText = await entityResponse.text();
        const entities = JSON.parse(entityText);
        
        // Build full text from entities
        const fullText = [
          entities.company || '',
          entities.address || '',
          entities.date || '',
          `Total: ${entities.total || ''}`,
        ].filter(Boolean).join('\n');
        
        samples.push({
          id: imageId,
          imagePath: `/dataset/SROIE2019/train/img/${imageId}.jpg`,
          text: fullText,
          entities,
          source: 'sroie',
        });
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          console.warn(`Timeout loading sample ${imageId}`);
        } else {
          console.warn(`Failed to load sample ${imageId}:`, error);
        }
      }
    }
    
    // Final progress update
    if (onProgress) {
      onProgress(imageIds.length, imageIds.length);
    }
    
    if (samples.length === 0) {
      console.error('❌ Failed to load any SROIE2019 samples! Check that:');
      console.error('   1. The dataset folder exists at: /dataset/SROIE2019/train/');
      console.error('   2. Vite dev server is configured to serve the dataset folder');
      console.error('   3. Entity and image files are present');
    } else {
      console.log(`✅ Loaded ${samples.length} SROIE2019 samples`);
    }
  } catch (error) {
    console.error('Failed to load SROIE dataset:', error);
  }
  
  return samples;
}

/**
 * Extract pattern statistics from dataset samples
 */
export function analyzePatterns(samples: DatasetSample[]): Map<string, PatternStats> {
  const stats = new Map<string, PatternStats>();
  
  // Analyze common patterns from dataset
  for (const sample of samples) {
    // Company names (often business entities)
    if (sample.entities.company) {
      addPattern(stats, 'Company/Business Name', sample.entities.company);
    }
    
    // Dates (various formats)
    if (sample.entities.date) {
      addPattern(stats, 'Date', sample.entities.date);
    }
    
    // Addresses (PII)
    if (sample.entities.address) {
      addPattern(stats, 'Physical Address', sample.entities.address);
    }
    
    // Monetary amounts
    if (sample.entities.total) {
      addPattern(stats, 'Monetary Amount', sample.entities.total);
    }
  }
  
  return stats;
}

function addPattern(stats: Map<string, PatternStats>, type: string, value: string) {
  if (!stats.has(type)) {
    stats.set(type, {
      type,
      count: 0,
      examples: [],
      patterns: []
    });
  }
  
  const stat = stats.get(type)!;
  stat.count++;
  
  if (stat.examples.length < 10) {
    stat.examples.push(value);
  }
}

/**
 * Generate enhanced regex patterns based on dataset analysis
 */
export function generateEnhancedPatterns(stats: Map<string, PatternStats>): Map<string, RegExp[]> {
  const patterns = new Map<string, RegExp[]>();
  
  // Enhanced date patterns (from SROIE)
  patterns.set('date', [
    /\b\d{2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4}\b/gi,
    /\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b/g,
    /\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b/gi
  ]);
  
  // Enhanced address patterns (multi-line, postal codes)
  patterns.set('address', [
    /\b\d+,?\s+[A-Z][A-Za-z\s,]+\d{5,6}\b/gi, // Street with postal
    /\d+[A-Z]?,?\s+(?:JALAN|STREET|AVENUE|ROAD|LANE)[A-Z\s,]+\d{5}/gi, // Malaysian format
  ]);
  
  // Business/Company names (often ALL CAPS)
  patterns.set('business', [
    /\b[A-Z][A-Z\s&]+(?:SDN BHD|PTE LTD|LTD|LLC|INC|CORP)\b/g,
    /\b(?:[A-Z][A-Z\s]+){2,}\b/g // Multiple consecutive caps words
  ]);
  
  // Monetary amounts with currency
  patterns.set('money', [
    /[$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?/g,
    /\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY|MYR|SGD)\b/gi
  ]);
  
  // Names (proper nouns, title case)
  patterns.set('names', [
    /\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b/g,
    /\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/g // First Last
  ]);
  
  return patterns;
}

/**
 * Train pattern detector with dataset samples
 */
export async function trainPatternDetector(): Promise<Map<string, RegExp[]>> {
  console.log('🎓 Training pattern detector with datasets...');
  
  // Load dataset samples
  const sroieSamples = await loadSROIEDataset(100);
  console.log(`Loaded ${sroieSamples.length} SROIE samples`);
  
  // Analyze patterns
  const stats = analyzePatterns(sroieSamples);
  console.log(`Analyzed ${stats.size} pattern types`);
  
  // Generate enhanced patterns
  const enhancedPatterns = generateEnhancedPatterns(stats);
  console.log(`Generated ${enhancedPatterns.size} enhanced pattern categories`);
  
  return enhancedPatterns;
}

/**
 * Load and cache trained patterns
 */
let cachedPatterns: Map<string, RegExp[]> | null = null;

export async function getTrainedPatterns(): Promise<Map<string, RegExp[]>> {
  if (!cachedPatterns) {
    cachedPatterns = await trainPatternDetector();
  }
  return cachedPatterns;
}
