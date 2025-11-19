package com.app.wineclassifierapi.service;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.AbstractEmbeddingModel;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import javax.validation.constraints.NotNull;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class RosbertaEmbeddingModel extends AbstractEmbeddingModel {
    RestClient restClient;

    @Override
    public @NotNull EmbeddingResponse call(@NotNull EmbeddingRequest request) {
        var payload = Map.of(
                "inputs", "search_query: " + request.getInstructions().get(0),
                "parameters",
                Map.of("pooling_method", "cls", "normalize_embeddings", true)
        );

        List<Double> responseList = restClient.post()
                .contentType(MediaType.APPLICATION_JSON)
                .body(payload)
                .retrieve()
                .body(new ParameterizedTypeReference<>() {});

        float[] floats = new float[responseList.size()];
        for (int i = 0; i < responseList.size(); i++) {
            floats[i] = responseList.get(i).floatValue();
        }

        return new EmbeddingResponse(List.of(new Embedding(floats, 0)));
    }

    @Override
    public @NotNull float[] embed(@NotNull Document document) {
        return call(new EmbeddingRequest(List.of(document.getFormattedContent()), null))
                .getResults().stream().findFirst().orElseThrow(
                        () -> new RuntimeException("Something went wrong, please retry")
                ).getOutput();
    }
}
