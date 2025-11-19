package com.app.wineclassifierapi.mapper;

import com.app.wineclassifierapi.dto.WineDto;
import org.springframework.ai.document.Document;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Stream;

@Service
public class WineMapper {

    public Document toWineDocument(WineDto wine) {

        Map<String, Object> metadata = new HashMap<>();
        metadata.put("tags", Stream.builder()
                .add(wine.tags().wineType())
                .add(wine.tags().region())
                .build().toList()
        );

        Calendar calendar = new GregorianCalendar();

        calendar.set(Calendar.YEAR, Integer.parseInt(wine.tags().date()));
        calendar.set(Calendar.MONTH, Calendar.JANUARY);
        calendar.set(Calendar.DAY_OF_MONTH, 1);
        calendar.set(Calendar.HOUR_OF_DAY, 0);
        calendar.set(Calendar.MINUTE, 0);
        calendar.set(Calendar.SECOND, 0);
        calendar.set(Calendar.MILLISECOND, 0);

        long milliseconds = calendar.getTimeInMillis();
        metadata.put("timestamp", milliseconds);

        return Document.builder()
                .text(wine.description())
                .metadata(metadata)
                .build();
    }

    public WineDto toWineDto(Document wine) {

        Map<String, Object> metadata = wine.getMetadata();

        long milliseconds = Long.parseLong((String) metadata.get("timestamp"));
        Calendar calendar = Calendar.getInstance();
        calendar.setTimeInMillis(milliseconds);
        Integer year = calendar.get(Calendar.YEAR);

        WineDto.Tags tags = new WineDto.Tags(
                ((List<String>) metadata.get("tags")).get(1),
                ((List<String>) metadata.get("tags")).get(0),
                year.toString()
        );

        return new WineDto(
                wine.getText(),
                tags
        );
    }
}
