package com.app.wineclassifierapi.service;

import com.app.wineclassifierapi.dto.WineDto;
import com.app.wineclassifierapi.mapper.WineMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.qdrant.QdrantVectorStore;
import org.springframework.cache.Cache;
import org.springframework.cache.CacheManager;
import org.springframework.cache.annotation.CachePut;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class WineService {

    private final QdrantVectorStore qdVectorStore;
    private final WineMapper wineMapper;
    private final CacheManager cacheManager;

    public void saveWines(List<WineDto> wines) {

        List<Document> winesToSave = wines.stream()
                .map(wineMapper::toWineDocument)
                .toList();

        winesToSave.stream()
                .forEach(wine -> qdVectorStore.doAdd(List.of(wine)));
    }

    @CachePut(value="SIMILAR_WINES", key="#wineInfo + '_' + #limit")
    public List<WineDto> getSimilarWines(String wineInfo, Integer limit) {

        SearchRequest searchRequest = SearchRequest.builder()
                .query(wineInfo)
                .topK(limit)
                .build();

        return qdVectorStore.similaritySearch(searchRequest).stream()
                .map(wineMapper::toWineDto)
                .toList();
    }

    public List<WineDto> getSimilarWinesFromCache(String wineInfo, String limit) {

        Cache cachedWines = cacheManager.getCache("SIMILAR_WINES");
        Cache.ValueWrapper wines = cachedWines.get(wineInfo + "_" + limit);
        return wines != null ? (List<WineDto>) wines.get() : null;
    }
}
