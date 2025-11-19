package com.app.wineclassifierapi.dto;


public record WineDto (

    String description,
    Tags tags

) {

    public record Tags (

            String region,
            String wineType,
            String date
    ) {}
}
