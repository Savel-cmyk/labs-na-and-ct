FROM gradle:8.14-jdk17 AS build

WORKDIR /app/wine-classifier

COPY gradle gradle
COPY gradlew .
COPY build.gradle .
COPY settings.gradle .

RUN chmod +x ./gradlew && ./gradlew dependencies --no-daemon

COPY src src

RUN ./gradlew bootJar --no-daemon

FROM eclipse-temurin:17-jre AS runtime

WORKDIR /app/wine-classifier

COPY --from=build /app/wine-classifier/build/libs/*.jar wine-classifier-api.jar

EXPOSE $APP_PORT

ENTRYPOINT ["java", "-jar", "wine-classifier-api.jar"]
