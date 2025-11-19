package com.app.wineclassifierapi.config;

import com.app.wineclassifierapi.service.RosbertaEmbeddingModel;
import io.qdrant.client.QdrantClient;
import io.qdrant.client.QdrantGrpcClient;
import org.springframework.ai.vectorstore.qdrant.QdrantVectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

@Configuration
public class QDrantConfig {

    @Value("${spring.ai.vectorstore.qdrant.host:localhost}")
    private String qdrantHost;

    @Value("${spring.ai.vectorstore.qdrant.port:6334}")
    private int qdrantPort;

    @Value("${spring.ai.vectorstore.qdrant.collection-name:spring}")
    private String collectionName;

    @Value("${spring.ai.vectorstore.qdrant.api-key:}")
    private String apiKey;

    @Bean
    @Primary
    public QdrantClient qdrantClient() {
        QdrantGrpcClient.Builder builder = QdrantGrpcClient
                .newBuilder(qdrantHost, qdrantPort, false);
        if (apiKey != null && !apiKey.trim().isEmpty()) {
            builder.withApiKey(apiKey);
        }
        return new QdrantClient(builder.build());
    }

    @Bean
    @Primary
    public QdrantVectorStore qdVectorStore(QdrantClient qdrantClient,
                                           RosbertaEmbeddingModel rosbertaEmbeddingModel) {
        QdrantVectorStore voStore = QdrantVectorStore
                .builder(qdrantClient, rosbertaEmbeddingModel)
                .collectionName(collectionName)
                .initializeSchema(true)
                .build();

        return voStore;
    }
}
