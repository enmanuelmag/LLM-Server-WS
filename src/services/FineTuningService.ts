import OpenAI from 'openai';
import fs from 'fs/promises';
import path from 'path';
import { FINE_TUNING_DATASET } from '../data/fine-tuning-dataset.js';
import {
  FineTuningDataPoint,
  EmailClassificationRequest,
  EmailClassificationResult,
} from '../types/fine-tuning.js';
import { Logger } from '../utils/logger.js';

/**
 * FineTuningService - Comprehensive Fine-Tuning Implementation
 *
 * Esta clase demuestra el uso completo del SDK de OpenAI para fine-tuning,
 * cubriendo todos los aspectos necesarios para la certificación OpenAI Practitioner:
 *
 * 1. Preparación de datasets (formato JSONL)
 * 2. Subida de archivos de entrenamiento
 * 3. Creación y monitoreo de jobs de fine-tuning
 * 4. Evaluación y comparación de modelos
 * 5. Uso del modelo fine-tuned para predicciones
 * 6. Gestión del ciclo de vida del modelo
 */
export class FineTuningService {
  private client: OpenAI;
  private trainingFileName: string;
  private validationFileName: string;
  private modelName: string;

  constructor() {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY environment variable is required');
    }

    this.client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    this.trainingFileName = 'email-classification-training.jsonl';
    this.validationFileName = 'email-classification-validation.jsonl';
    this.modelName = 'email-classifier-v1';
  }

  /**
   * 1. PREPARACIÓN DEL DATASET
   * Convierte el dataset interno a formato JSONL requerido por OpenAI
   */
  async prepareDataset(): Promise<{
    trainingPath: string;
    validationPath: string;
  }> {
    Logger.info('🔧 Preparando dataset para fine-tuning...');

    // Dividir dataset en entrenamiento (80%) y validación (20%)
    const shuffledData = [...FINE_TUNING_DATASET].sort(
      () => 0.5 - Math.random()
    );
    const splitIndex = Math.floor(shuffledData.length * 0.8);

    const trainingData = shuffledData.slice(0, splitIndex);
    const validationData = shuffledData.slice(splitIndex);

    Logger.info(
      `📊 Dataset dividido: ${trainingData.length} entrenamiento, ${validationData.length} validación`
    );

    // Crear archivos JSONL
    const trainingPath = path.join(
      process.cwd(),
      'uploads',
      this.trainingFileName
    );
    const validationPath = path.join(
      process.cwd(),
      'uploads',
      this.validationFileName
    );

    await this.writeJSONL(trainingData, trainingPath);
    await this.writeJSONL(validationData, validationPath);

    // Validar formato del dataset
    await this.validateDataset(trainingData);

    Logger.info('✅ Dataset preparado correctamente');
    return { trainingPath, validationPath };
  }

  /**
   * Escribe datos en formato JSONL
   */
  private async writeJSONL(
    data: FineTuningDataPoint[],
    filePath: string
  ): Promise<void> {
    const jsonlContent = data.map((item) => JSON.stringify(item)).join('\n');

    await fs.writeFile(filePath, jsonlContent, 'utf-8');
    Logger.info(`📝 Archivo JSONL creado: ${filePath}`);
  }

  /**
   * Valida que el dataset cumpla con los requisitos de OpenAI
   */
  private async validateDataset(data: FineTuningDataPoint[]): Promise<void> {
    Logger.info('🔍 Validando formato del dataset...');

    for (let i = 0; i < data.length; i++) {
      const item = data[i];

      // Verificar estructura de mensajes
      if (!item.messages || !Array.isArray(item.messages)) {
        throw new Error(`Item ${i}: messages debe ser un array`);
      }

      // Verificar roles requeridos
      const roles = item.messages.map((m) => m.role);
      if (
        !roles.includes('system') ||
        !roles.includes('user') ||
        !roles.includes('assistant')
      ) {
        throw new Error(
          `Item ${i}: debe incluir roles system, user y assistant`
        );
      }

      // Verificar longitud de contenido
      for (const message of item.messages) {
        if (message.content.length > 4000) {
          Logger.warn(
            `Item ${i}: mensaje muy largo (${message.content.length} chars)`
          );
        }
      }
    }

    Logger.info('✅ Dataset válido');
  }

  /**
   * 2. SUBIDA DE ARCHIVOS
   * Sube los archivos de entrenamiento a OpenAI
   */
  async uploadTrainingFiles(): Promise<{
    trainingFileId: string;
    validationFileId?: string;
  }> {
    Logger.info('📤 Subiendo archivos de entrenamiento...');

    const { trainingPath, validationPath } = await this.prepareDataset();

    // Subir archivo de entrenamiento
    const trainingFileBuffer = await fs.readFile(trainingPath);
    //! Usar el módulo files para crear y subir el archivo
    //! La variable debe ser trainingFile
    const trainingFile: any = {}; // Reemplazar con la llamada real
    Logger.info(`✅ Archivo de entrenamiento subido: ${trainingFile.id}`);

    // Subir archivo de validación (opcional pero recomendado)
    let validationFileId: string | undefined;
    try {
      const validationFileBuffer = await fs.readFile(validationPath);
      //! Usar el módulo files para crear y subir el archivo
      // const validationFile = await this.client.files.create({
      const validationFile: any = {}; // Reemplazar con la llamada real

      validationFileId = validationFile.id;
      Logger.info(`✅ Archivo de validación subido: ${validationFile.id}`);
    } catch (error) {
      Logger.warn('⚠️ No se pudo subir archivo de validación:', error);
    }

    return {
      trainingFileId: trainingFile.id,
      validationFileId,
    };
  }

  /**
   * 3. CREACIÓN DEL JOB DE FINE-TUNING
   * Crea y configura el job de fine-tuning con todos los parámetros importantes
   */
  async createFineTuningJob(): Promise<string> {
    Logger.info('🚀 Iniciando job de fine-tuning...');

    const { trainingFileId, validationFileId } =
      await this.uploadTrainingFiles();

    //! Crear el job, pasándole ambos archivos IDs
    //! El modelo base debe ser gpt-3.5-turbo-1106
    //! El sufijo debe ser this.modelName
    //! Hyper-parámetros recomendados: n_epochs=3, batch_size=1, learning_rate_multiplier=0.1
    const job: any = {}; // Reemplazar con la llamada real

    Logger.info(`✅ Job de fine-tuning creado: ${job.id}`);
    Logger.info(`📋 Configuración:`, {
      model: job.model,
      trainingFile: job.training_file,
      validationFile: job.validation_file,
      hyperparameters: job.hyperparameters,
    });

    return job.id;
  }

  async classifyEmailWithFineTunedModel(
    modelId: string,
    emailData: EmailClassificationRequest
  ): Promise<EmailClassificationResult> {
    Logger.info(`🤖 Clasificando email con modelo: ${modelId}`);

    const systemPrompt =
      'Clasifica si este email contiene información de transacciones financieras y extrae detalles relevantes.';
    const userPrompt = `Subject: ${emailData.emailSubject}\nBody: ${emailData.emailBody}`;

    const completion = await this.client.chat.completions.create({
      model: modelId,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      temperature: 0.1, // Baja temperatura para consistencia
      max_tokens: 500,
    });

    const response = completion.choices[0].message?.content;
    if (!response) {
      throw new Error('No se recibió respuesta del modelo');
    }

    try {
      const result: EmailClassificationResult = JSON.parse(response);
      Logger.info('✅ Clasificación completada:', result);
      return result;
    } catch (error) {
      Logger.error('❌ Error parseando respuesta del modelo:', response);
      throw new Error('Formato de respuesta inválido del modelo fine-tuned');
    }
  }

  async listFineTunedModels(): Promise<void> {
    Logger.info('📋 Listando modelos fine-tuned...');

    const models = await this.client.models.list();
    const fineTunedModels = models.data.filter(
      (model) => model.id.includes('ft:') && model.owned_by === 'user'
    );

    if (fineTunedModels.length === 0) {
      Logger.info('❌ No se encontraron modelos fine-tuned');
      return;
    }

    Logger.info(
      `✅ Se encontraron ${fineTunedModels.length} modelos fine-tuned:`
    );

    for (const model of fineTunedModels) {
      Logger.info(`  🎯 ${model.id}`);
      Logger.info(
        `     Creado: ${new Date(model.created * 1000).toISOString()}`
      );
      Logger.info(`     Propietario: ${model.owned_by}`);
    }
  }

  async executeCompleteWorkflow(): Promise<string> {
    Logger.info('🚀 INICIANDO WORKFLOW COMPLETO DE FINE-TUNING');
    Logger.info('='.repeat(60));

    try {
      // Paso 1: Preparar y validar dataset
      Logger.info('\n🔧 PASO 1: Preparación del Dataset');
      await this.prepareDataset();

      // Paso 2: Crear job de fine-tuning
      Logger.info('\n🚀 PASO 2: Creación del Job');
      const jobId = await this.createFineTuningJob();

      // Paso 4: Probar modelo
      Logger.info('\n🧪 PASO 4: Prueba del Modelo');
      const testEmail: EmailClassificationRequest = {
        emailSubject: 'Invoice #12345 - Payment Due',
        emailBody:
          'Your invoice for $1,500.00 is now due. Please process payment within 30 days.',
      };

      const result = await this.classifyEmailWithFineTunedModel(
        'modelId',
        testEmail
      );
      Logger.info('📊 Resultado de prueba:', result);

      // Paso 5: Comparar con modelo base
      Logger.info('\n📈 PASO 5: Comparación de Modelos');

      // Paso 6: Listar todos los modelos
      Logger.info('\n📋 PASO 6: Gestión de Modelos');
      await this.listFineTunedModels();

      Logger.info('\n✅ WORKFLOW COMPLETO FINALIZADO');
      Logger.info('='.repeat(60));

      return 'modelId';
    } catch (error) {
      Logger.error('❌ Error en el workflow:', error);
      throw error;
    }
  }
}
