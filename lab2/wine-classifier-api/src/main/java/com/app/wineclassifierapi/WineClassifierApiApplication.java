package com.app.wineclassifierapi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;

@SpringBootApplication
@EnableCaching
public class WineClassifierApiApplication {

    public static void main(String[] args) {
        SpringApplication.run(WineClassifierApiApplication.class, args);
    }

}
