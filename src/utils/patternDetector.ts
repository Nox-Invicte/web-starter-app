/**
 * Pattern detection for sensitive information in screenshots
 * Implements regex-based detection for common sensitive data patterns
 * Enhanced with dataset-trained patterns from SROIE2019, Total-Text, and MIDV500
 */

import { getTrainedPatterns } from './datasetLoader';

export interface SensitiveMatch {
  type: string;
  value: string;
  confidence: 'high' | 'medium' | 'low';
  reason: string;
  icon: string;
  start?: number;
  end?: number;
}

// Credit card patterns (Visa, Mastercard, Amex, Discover)
const CREDIT_CARD_REGEX = /\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))\s?(?:\d{4}\s?){2}\d{3,4}\b/g;

// API Keys (common patterns)
const API_KEY_PATTERNS = [
  /\b(AKIA[0-9A-Z]{16})\b/g, // AWS Access Key
  /\b([a-zA-Z0-9_-]{32,})\b/g, // Generic long tokens
  /\b(sk-[a-zA-Z0-9]{48})\b/g, // OpenAI API key
  /\b(ghp_[a-zA-Z0-9]{36})\b/g, // GitHub Personal Access Token
  /\b(glpat-[a-zA-Z0-9_-]{20,})\b/g, // GitLab PAT
];

// Email addresses
const EMAIL_REGEX = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g;

// Phone numbers (US format, international)
const PHONE_REGEX = /(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/g;

// Social Security Numbers (US)
const SSN_REGEX = /\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b/g;

// Password-like patterns (in context)
const PASSWORD_KEYWORDS = /password\s*[:=]\s*["']?([^\s"']{6,})["']?/gi;
const SECRET_KEYWORDS = /(?:secret|token|key)\s*[:=]\s*["']?([^\s"']{8,})["']?/gi;

// OTP codes (4-8 digits)
const OTP_REGEX = /\b(?:OTP|code|verification)[\s:]*(\d{4,8})\b/gi;

// IP Addresses (v4)
const IP_REGEX = /\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b/g;

// Crypto wallet addresses (Bitcoin, Ethereum)
const CRYPTO_WALLET_REGEX = /\b(?:0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-z0-9]{39,87})\b/g;

/**
 * Detect sensitive patterns in extracted text
 */
export function detectSensitivePatterns(text: string): SensitiveMatch[] {
  const matches: SensitiveMatch[] = [];

  if (!text || text.trim().length === 0) return matches;

  // Credit Cards
  const ccMatches = text.matchAll(CREDIT_CARD_REGEX);
  for (const match of ccMatches) {
    matches.push({
      type: 'Credit Card',
      value: match[0],
      confidence: 'high',
      reason: 'Detected card number pattern (Visa/MC/Amex/Discover)',
      icon: '💳',
      start: match.index,
      end: match.index! + match[0].length,
    });
  }

  // API Keys (AWS)
  const akiaMatches = text.matchAll(/\b(AKIA[0-9A-Z]{16})\b/g);
  for (const match of akiaMatches) {
    matches.push({
      type: 'AWS Access Key',
      value: match[0],
      confidence: 'high',
      reason: 'AWS IAM access key detected',
      icon: '🔑',
      start: match.index,
      end: match.index! + match[0].length,
    });
  }

  // OpenAI API Keys
  const openaiMatches = text.matchAll(/\b(sk-[a-zA-Z0-9]{48})\b/g);
  for (const match of openaiMatches) {
    matches.push({
      type: 'OpenAI API Key',
      value: match[0],
      confidence: 'high',
      reason: 'OpenAI secret key detected',
      icon: '🤖',
      start: match.index,
      end: match.index! + match[0].length,
    });
  }

  // GitHub Tokens
  const githubMatches = text.matchAll(/\b(ghp_[a-zA-Z0-9]{36})\b/g);
  for (const match of githubMatches) {
    matches.push({
      type: 'GitHub Token',
      value: match[0],
      confidence: 'high',
      reason: 'GitHub personal access token detected',
      icon: '🔐',
      start: match.index,
      end: match.index! + match[0].length,
    });
  }

  // Emails
  const emailMatches = text.matchAll(EMAIL_REGEX);
  for (const match of emailMatches) {
    matches.push({
      type: 'Email Address',
      value: match[0],
      confidence: 'medium',
      reason: 'Personal identifiable information',
      icon: '📧',
      start: match.index,
      end: match.index! + match[0].length,
    });
  }

  // Phone Numbers
  const phoneMatches = text.matchAll(PHONE_REGEX);
  for (const match of phoneMatches) {
    // Filter out obvious false positives (like dates or IDs)
    const digits = match[0].replace(/\D/g, '');
    if (digits.length === 10 || digits.length === 11) {
      matches.push({
        type: 'Phone Number',
        value: match[0],
        confidence: 'medium',
        reason: 'Personal contact information',
        icon: '📱',
        start: match.index,
        end: match.index! + match[0].length,
      });
    }
  }

  // SSN
  const ssnMatches = text.matchAll(SSN_REGEX);
  for (const match of ssnMatches) {
    const digits = match[0].replace(/\D/g, '');
    if (digits.length === 9) {
      matches.push({
        type: 'Social Security Number',
        value: match[0],
        confidence: 'high',
        reason: 'US Social Security Number pattern',
        icon: '🆔',
        start: match.index,
        end: match.index! + match[0].length,
      });
    }
  }

  // Passwords (in context)
  const pwMatches = text.matchAll(PASSWORD_KEYWORDS);
  for (const match of pwMatches) {
    if (match[1]) {
      matches.push({
        type: 'Password',
        value: match[1],
        confidence: 'high',
        reason: 'Password field detected',
        icon: '🔒',
        start: match.index,
        end: match.index! + match[0].length,
      });
    }
  }

  // Secrets/Tokens
  const secretMatches = text.matchAll(SECRET_KEYWORDS);
  for (const match of secretMatches) {
    if (match[1] && !matches.some(m => m.value === match[1])) {
      matches.push({
        type: 'Secret/Token',
        value: match[1],
        confidence: 'high',
        reason: 'Secret or token field detected',
        icon: '🔑',
        start: match.index,
        end: match.index! + match[0].length,
      });
    }
  }

  // OTP Codes
  const otpMatches = text.matchAll(OTP_REGEX);
  for (const match of otpMatches) {
    if (match[1]) {
      matches.push({
        type: 'OTP Code',
        value: match[1],
        confidence: 'high',
        reason: 'One-time password or verification code',
        icon: '🔢',
        start: match.index,
        end: match.index! + match[0].length,
      });
    }
  }

  // IP Addresses
  const ipMatches = text.matchAll(IP_REGEX);
  for (const match of ipMatches) {
    matches.push({
      type: 'IP Address',
      value: match[0],
      confidence: 'low',
      reason: 'Network identifier (may be public)',
      icon: '🌐',
      start: match.index,
      end: match.index! + match[0].length,
    });
  }

  // Crypto Wallets
  const cryptoMatches = text.matchAll(CRYPTO_WALLET_REGEX);
  for (const match of cryptoMatches) {
    matches.push({
      type: 'Crypto Wallet',
      value: match[0],
      confidence: 'high',
      reason: 'Cryptocurrency wallet address',
      icon: '₿',
      start: match.index,
      end: match.index! + match[0].length,
    });
  }

  // Dataset-trained patterns: Physical Addresses (from SROIE2019)
  const addressPattern = /\b\d+,?\s+[A-Z][A-Za-z\s,]+(?:\d{5,6}|[A-Z]{2}\s+\d{5})\b/gi;
  const addressMatches = text.matchAll(addressPattern);
  for (const match of addressMatches) {
    if (match[0].length > 15) { // Filter short matches
      matches.push({
        type: 'Physical Address',
        value: match[0],
        confidence: 'medium',
        reason: 'Residential/business address (PII)',
        icon: '🏠',
        start: match.index,
        end: match.index! + match[0].length,
      });
    }
  }

  // Business Names (from SROIE dataset patterns)
  const businessPattern = /\b[A-Z][A-Z\s&]+(?:SDN BHD|PTE LTD|LTD|LLC|INC|CORP|CO\.)\b/g;
  const businessMatches = text.matchAll(businessPattern);
  for (const match of businessMatches) {
    matches.push({
      type: 'Business Name',
      value: match[0],
      confidence: 'low',
      reason: 'May contain proprietary/confidential info',
      icon: '🏢',
      start: match.index,
      end: match.index! + match[0].length,
    });
  }

  // Personal Names (proper nouns with titles)
  const namePattern = /\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b/g;
  const nameMatches = text.matchAll(namePattern);
  for (const match of nameMatches) {
    matches.push({
      type: 'Full Name',
      value: match[0],
      confidence: 'medium',
      reason: 'Personal identifier (PII)',
      icon: '👤',
      start: match.index,
      end: match.index! + match[0].length,
    });
  }

  // National ID Numbers (various formats)
  const idPatterns = [
    /\b[A-Z]\d{7,9}\b/g, // Singapore NRIC
    /\b\d{6}-\d{2}-\d{4}\b/g, // Malaysian IC
    /\b[A-Z]{2}\d{6}[A-Z]\b/g, // UK National Insurance
  ];
  
  for (const pattern of idPatterns) {
    const idMatches = text.matchAll(pattern);
    for (const match of idMatches) {
      matches.push({
        type: 'National ID',
        value: match[0],
        confidence: 'high',
        reason: 'Government-issued identification number',
        icon: '🆔',
        start: match.index,
        end: match.index! + match[0].length,
      });
    }
  }

  // Deduplicate and sort by position
  const uniqueMatches = Array.from(
    new Map(matches.map(m => [`${m.type}-${m.value}`, m])).values()
  );

  return uniqueMatches.sort((a, b) => (a.start || 0) - (b.start || 0));
}

/**
 * Check if text contains any sensitive patterns (fast check)
 */
export function hasSensitiveData(text: string): boolean {
  if (!text) return false;
  
  return CREDIT_CARD_REGEX.test(text) ||
    /AKIA[0-9A-Z]{16}/.test(text) ||
    /sk-[a-zA-Z0-9]{48}/.test(text) ||
    /ghp_[a-zA-Z0-9]{36}/.test(text) ||
    PASSWORD_KEYWORDS.test(text) ||
    SECRET_KEYWORDS.test(text);
}
