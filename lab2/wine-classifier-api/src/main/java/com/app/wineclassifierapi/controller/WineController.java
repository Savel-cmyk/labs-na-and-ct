package com.app.wineclassifierapi.controller;

import com.app.wineclassifierapi.dto.WineDto;
import com.app.wineclassifierapi.service.WineService;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.document.Document;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/v1/wine")
public class WineController {

    private final WineService wineService;

    public WineController(WineService wineService) {
        this.wineService = wineService;
    }

    @PostMapping
    public ResponseEntity saveWines(@RequestBody List<WineDto> wines) {
        wineService.saveWines(wines);
        return ResponseEntity.ok().build();
    }

    @GetMapping
    public ResponseEntity<List<WineDto>> getSimilarWines(
            @RequestBody String wineInfo,
            @RequestParam(defaultValue = "3") int limit
    ) {
        return ResponseEntity.ok(wineService.getSimilarWines(wineInfo, limit));
    }

    @GetMapping("/cache")
    public ResponseEntity<List<WineDto>> getCachedSimilarWines(
            @RequestBody String wineInfo,
            @RequestParam(defaultValue = "3") String limit
    ) {
        return ResponseEntity.ok(wineService.getSimilarWinesFromCache(wineInfo, limit));
    }
}
