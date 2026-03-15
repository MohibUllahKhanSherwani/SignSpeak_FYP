package com.example.kotlinfrontend

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import java.io.IOException

private val Context.backendSettingsDataStore by preferencesDataStore(name = "backend_settings")

class BackendSettingsRepository(private val context: Context) {
    private object Keys {
        val backendBaseUrl = stringPreferencesKey("backend_base_url")
    }

    val backendBaseUrlFlow: Flow<String> = context.backendSettingsDataStore.data
        .catch { error ->
            if (error is IOException) {
                emit(emptyPreferences())
            } else {
                throw error
            }
        }
        .map { preferences ->
            preferences[Keys.backendBaseUrl].orEmpty()
        }

    suspend fun saveBackendBaseUrl(baseUrl: String) {
        context.backendSettingsDataStore.edit { preferences ->
            if (baseUrl.isBlank()) {
                preferences.remove(Keys.backendBaseUrl)
            } else {
                preferences[Keys.backendBaseUrl] = baseUrl
            }
        }
    }
}
