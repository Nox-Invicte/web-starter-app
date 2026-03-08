/**
 * Image preprocessing utilities for improved OCR accuracy
 * Applies various enhancement techniques to optimize text extraction
 */

/**
 * Preprocessing options
 */
export interface PreprocessOptions {
  // Enhance contrast for better text visibility
  enhanceContrast?: boolean;
  // Convert to grayscale (often improves OCR)
  grayscale?: boolean;
  // Apply sharpening filter
  sharpen?: boolean;
  // Increase brightness
  brightness?: number; // -100 to 100
  // Adaptive thresholding for binarization
  adaptiveThreshold?: boolean;
  // Denoise the image
  denoise?: boolean;
  // Target size for processing (default: 512)
  targetSize?: number;
}

/**
 * Preprocess image for optimal OCR performance
 */
export function preprocessImage(
  imageData: ImageData,
  options: PreprocessOptions = {}
): ImageData {
  const {
    enhanceContrast = true,
    grayscale = true,
    sharpen = true,
    brightness = 10,
    denoise = false,
    adaptiveThreshold = false,
  } = options;

  let processed = imageData;

  // Apply brightness adjustment
  if (brightness !== 0) {
    processed = adjustBrightness(processed, brightness);
  }

  // Convert to grayscale
  if (grayscale) {
    processed = toGrayscale(processed);
  }

  // Enhance contrast
  if (enhanceContrast) {
    processed = enhanceImageContrast(processed);
  }

  // Sharpen
  if (sharpen) {
    processed = sharpenImage(processed);
  }

  // Denoise
  if (denoise) {
    processed = denoiseImage(processed);
  }

  // Adaptive thresholding (binarization)
  if (adaptiveThreshold) {
    processed = applyAdaptiveThreshold(processed);
  }

  return processed;
}

/**
 * Adjust image brightness
 */
function adjustBrightness(imageData: ImageData, amount: number): ImageData {
  const data = new Uint8ClampedArray(imageData.data);
  
  for (let i = 0; i < data.length; i += 4) {
    data[i] = clamp(data[i] + amount, 0, 255);     // R
    data[i + 1] = clamp(data[i + 1] + amount, 0, 255); // G
    data[i + 2] = clamp(data[i + 2] + amount, 0, 255); // B
  }
  
  return new ImageData(data, imageData.width, imageData.height);
}

/**
 * Convert to grayscale using luminosity method
 */
function toGrayscale(imageData: ImageData): ImageData {
  const data = new Uint8ClampedArray(imageData.data);
  
  for (let i = 0; i < data.length; i += 4) {
    // Luminosity method (more accurate perception)
    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    data[i] = gray;
    data[i + 1] = gray;
    data[i + 2] = gray;
  }
  
  return new ImageData(data, imageData.width, imageData.height);
}

/**
 * Enhance contrast using histogram stretching
 */
function enhanceImageContrast(imageData: ImageData): ImageData {
  const data = new Uint8ClampedArray(imageData.data);
  
  // Find min and max values
  let min = 255, max = 0;
  for (let i = 0; i < data.length; i += 4) {
    const gray = data[i]; // Assumes grayscale or use all channels
    min = Math.min(min, gray);
    max = Math.max(max, gray);
  }
  
  // Stretch histogram
  const range = max - min;
  if (range > 0) {
    for (let i = 0; i < data.length; i += 4) {
      const normalized = ((data[i] - min) / range) * 255;
      data[i] = normalized;
      data[i + 1] = normalized;
      data[i + 2] = normalized;
    }
  }
  
  return new ImageData(data, imageData.width, imageData.height);
}

/**
 * Sharpen image using convolution kernel
 */
function sharpenImage(imageData: ImageData): ImageData {
  // Sharpening kernel
  const kernel = [
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
  ];
  
  return applyConvolution(imageData, kernel, 1);
}

/**
 * Denoise using simple averaging filter
 */
function denoiseImage(imageData: ImageData): ImageData {
  // 3x3 averaging kernel
  const kernel = [
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
  ];
  
  return applyConvolution(imageData, kernel, 9);
}

/**
 * Apply convolution kernel to image
 */
function applyConvolution(
  imageData: ImageData,
  kernel: number[],
  divisor: number
): ImageData {
  const { width, height, data } = imageData;
  const output = new Uint8ClampedArray(data.length);
  
  const kSize = Math.sqrt(kernel.length);
  const kHalf = Math.floor(kSize / 2);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = 0, g = 0, b = 0;
      
      for (let ky = 0; ky < kSize; ky++) {
        for (let kx = 0; kx < kSize; kx++) {
          const px = clamp(x + kx - kHalf, 0, width - 1);
          const py = clamp(y + ky - kHalf, 0, height - 1);
          const pIdx = (py * width + px) * 4;
          const kVal = kernel[ky * kSize + kx];
          
          r += data[pIdx] * kVal;
          g += data[pIdx + 1] * kVal;
          b += data[pIdx + 2] * kVal;
        }
      }
      
      const idx = (y * width + x) * 4;
      output[idx] = clamp(r / divisor, 0, 255);
      output[idx + 1] = clamp(g / divisor, 0, 255);
      output[idx + 2] = clamp(b / divisor, 0, 255);
      output[idx + 3] = data[idx + 3]; // Alpha
    }
  }
  
  return new ImageData(output, width, height);
}

/**
 * Apply adaptive thresholding for binarization
 * Better than global thresholding for varying lighting
 */
function applyAdaptiveThreshold(imageData: ImageData): ImageData {
  const { width, height, data } = imageData;
  const output = new Uint8ClampedArray(data.length);
  const windowSize = 15;
  const C = 10; // Constant subtracted from mean
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Calculate local mean in window
      let sum = 0, count = 0;
      
      for (let wy = Math.max(0, y - windowSize); wy < Math.min(height, y + windowSize); wy++) {
        for (let wx = Math.max(0, x - windowSize); wx < Math.min(width, x + windowSize); wx++) {
          const idx = (wy * width + wx) * 4;
          sum += data[idx];
          count++;
        }
      }
      
      const mean = sum / count;
      const idx = (y * width + x) * 4;
      const pixel = data[idx];
      const threshold = pixel > (mean - C) ? 255 : 0;
      
      output[idx] = threshold;
      output[idx + 1] = threshold;
      output[idx + 2] = threshold;
      output[idx + 3] = data[idx + 3];
    }
  }
  
  return new ImageData(output, width, height);
}

/**
 * Utility: Clamp value between min and max
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Resize image while maintaining aspect ratio
 */
export function resizeImage(
  canvas: HTMLCanvasElement,
  targetSize: number = 512
): { width: number; height: number; canvas: HTMLCanvasElement } {
  const scale = Math.min(targetSize / canvas.width, targetSize / canvas.height);
  const newWidth = Math.floor(canvas.width * scale);
  const newHeight = Math.floor(canvas.height * scale);
  
  const resizedCanvas = document.createElement('canvas');
  resizedCanvas.width = newWidth;
  resizedCanvas.height = newHeight;
  
  const ctx = resizedCanvas.getContext('2d')!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(canvas, 0, 0, newWidth, newHeight);
  
  return { width: newWidth, height: newHeight, canvas: resizedCanvas };
}

/**
 * Auto-detect and apply optimal preprocessing
 */
export function autoPreprocess(imageData: ImageData): ImageData {
  // Analyze image characteristics
  const brightness = calculateAverageBrightness(imageData);
  const contrast = calculateContrast(imageData);
  
  const options: PreprocessOptions = {
    grayscale: true,
    enhanceContrast: contrast < 50, // Low contrast images
    sharpen: true,
    brightness: brightness < 100 ? 20 : 0, // Dark images
    denoise: false,
    adaptiveThreshold: false,
  };
  
  return preprocessImage(imageData, options);
}

/**
 * Calculate average brightness
 */
function calculateAverageBrightness(imageData: ImageData): number {
  let sum = 0;
  for (let i = 0; i < imageData.data.length; i += 4) {
    const gray = 0.299 * imageData.data[i] + 
                 0.587 * imageData.data[i + 1] + 
                 0.114 * imageData.data[i + 2];
    sum += gray;
  }
  return sum / (imageData.data.length / 4);
}

/**
 * Calculate contrast (standard deviation)
 */
function calculateContrast(imageData: ImageData): number {
  const brightness = calculateAverageBrightness(imageData);
  let sumSquares = 0;
  
  for (let i = 0; i < imageData.data.length; i += 4) {
    const gray = 0.299 * imageData.data[i] + 
                 0.587 * imageData.data[i + 1] + 
                 0.114 * imageData.data[i + 2];
    sumSquares += Math.pow(gray - brightness, 2);
  }
  
  return Math.sqrt(sumSquares / (imageData.data.length / 4));
}
