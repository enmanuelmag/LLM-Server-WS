import OpenAI from 'openai';
import {
  VectorItem,
  VectorEmbedItem,
  VectorStoreQuery,
  VectorSearchResult,
} from '../types/rag';
import { config } from '../config';
import { Logger } from '../utils/logger';

export class VectorStoreService {
  private openai: OpenAI;
  private items: VectorEmbedItem[] = [];
  private isInitialized = false;
  private sourceItems: VectorItem[];

  constructor(items: VectorItem[]) {
    this.openai = new OpenAI({
      apiKey: config.openai.apiKey,
    });
    this.sourceItems = items;
  }

  /**
   * Initialize vector store by creating embeddings for all items
   * En producción, esto se haría offline y se guardaría en una DB vectorial
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    Logger.info('🔄 Initializing vector store with provided items...');

    try {
      const itemsWithEmbeddings = await Promise.all(
        this.sourceItems.map(async (item) => {
          const embedding = await this.createEmbedding(item.content);
          return {
            ...item,
            embedding,
          } as VectorEmbedItem;
        })
      );

      this.items = itemsWithEmbeddings;
      this.isInitialized = true;

      Logger.success(
        `✅ Vector store initialized with ${this.items.length} items`
      );
    } catch (error) {
      Logger.error('❌ Failed to initialize vector store:', error);
      throw error;
    }
  }

  /**
   * Create embedding for text using OpenAI's embedding model
   */
  private async createEmbedding(text: string): Promise<number[]> {
    //! Crear el embedding usando OpenAI, con el módulo "embeddings"
    //! La variable debe ser "response"
    const response: any = {};

    return response.data[0].embedding;
  }

  /**
   * Search for similar items based on query
   */
  async searchSimilar(
    query: string,
    limit: number = 3,
    threshold: number = 0.7
  ): Promise<VectorSearchResult[]> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    Logger.debug('🔍 Searching vector store for:', query);

    //! Create embedding for the query
    const queryEmbedding: any = [];

    // Calculate similarity with all items
    const similarities = this.items.map((item) => ({
      item,
      similarity: this.cosineSimilarity(queryEmbedding, item.embedding),
    }));

    // Filter by threshold and sort by similarity
    const results = similarities
      .filter((result) => result.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);

    Logger.debug(
      `📊 Found ${results.length} relevant items (threshold: ${threshold})`
    );

    return results;
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  private cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (vecA.length !== vecB.length) {
      throw new Error('Vectors must have the same dimension');
    }

    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));

    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0;
    }

    return dotProduct / (magnitudeA * magnitudeB);
  }

  /**
   * Get all items (for debugging)
   */
  getAllItems(): VectorEmbedItem[] {
    return this.items;
  }

  /**
   * Get item by ID
   */
  getItemById(id: string): VectorEmbedItem | undefined {
    return this.items.find((item) => item.id === id);
  }
}
